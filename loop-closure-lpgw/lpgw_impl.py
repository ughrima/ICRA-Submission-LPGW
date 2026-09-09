import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import ot
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')


class ImprovedLPGWLoopClosure:
    """
    Improved LPGW-based loop closure detection with better precision
    """
    
    def __init__(self, segment_length=3.0, fps=10, stride=1.0, 
                 lambdaa=0.5, partial=True, downsampling=True):
        self.segment_length = segment_length
        self.fps = fps
        self.stride = stride
        self.lambdaa = lambdaa
        self.partial = partial
        self.downsampling = downsampling
        self.reference_segment = None
        self.distance_matrix = None
        
    def normalize_point_cloud(self, points):
        """Normalize point cloud with robust scaling"""
        scaler = StandardScaler()
        return scaler.fit_transform(points)
    
    def select_robust_reference(self, segments):
        """Select reference segment using robust statistical measures"""
        if not segments:
            raise ValueError("Segments list is empty.")
        
        # Calculate multiple criteria for reference selection
        sizes = [seg.shape[0] for seg in segments]
        complexities = []
        densities = []
        
        for seg in segments:
            # Measure complexity using variance of distances
            distances = cdist(seg, seg).flatten()
            complexities.append(np.var(distances))
            
            # Measure density (points per unit volume)
            bbox_size = np.max(seg, axis=0) - np.min(seg, axis=0)
            volume = np.prod(bbox_size) if np.all(bbox_size > 0) else 1.0
            densities.append(len(seg) / volume)
        
        # Normalize criteria
        complexities = np.array(complexities) / (np.max(complexities) + 1e-8)
        densities = np.array(densities) / (np.max(densities) + 1e-8)
        size_scores = 1 - np.abs(np.array(sizes) - np.median(sizes)) / np.median(sizes)
        
        # Combined score (weighted average)
        scores = 0.4 * size_scores + 0.3 * complexities + 0.3 * densities
        
        return segments[np.argmax(scores)]
    
    def compute_enhanced_lgw_embedding(self, X, Y, p=None, q=None):
        """LPGW embedding following paper Equation 12 and numerical implementation Equation 22.
        
        Returns:
            k_gamma: Embedding matrix (n x n) = D_X - D_Y_proj
            q_tilde: Transported mass distribution (n,) = gamma.sum(axis=1)
            gamma_c_abs: Mass creation term |γ_c|
            transported_mass: |γ_X| = q_tilde.sum()
            nu_sq: |ν|² = |γ_c| + |γ_X|² (for use in distance computation)
        """
        if X.shape[0] == 0 or Y.shape[0] == 0:
            raise ValueError("Input point clouds cannot be empty.")
        
        # Normalize point clouds (as in paper)
        X_norm = self.normalize_point_cloud(X)
        Y_norm = self.normalize_point_cloud(Y)
        
        # Distance matrices (squared Euclidean) - Equation 22 in paper
        D_X = cdist(X_norm, X_norm, metric='euclidean') ** 2
        D_Y = cdist(Y_norm, Y_norm, metric='euclidean') ** 2
        
        # Normalize by max (as in paper, not percentile)
        D_X_max = D_X.max() + 1e-8
        D_Y_max = D_Y.max() + 1e-8
        D_X_norm = D_X / D_X_max
        D_Y_norm = D_Y / D_Y_max
        
        # Mass distributions
        n, m = X.shape[0], Y.shape[0]
        p = np.ones(n) / n if p is None else np.array(p) / np.sum(p)
        q = np.ones(m) / m if q is None else np.array(q) / np.sum(q)
        
        try:
            if self.partial:
                # Use partial_gromov_wasserstein (not entropic) as in paper
                # reg_m is solver regularization (separate from λ)
                total_mass = min(p.sum(), q.sum())
                gamma = ot.gromov.partial_gromov_wasserstein(
                    D_X_norm, D_Y_norm, p, q,
                    loss_fun='square_loss', m=total_mass, reg_m=1e-3, verbose=False
                )
            else:
                gamma = ot.gromov.gromov_wasserstein(
                    D_X_norm, D_Y_norm, p, q, 'square_loss', verbose=False
                )
        except Exception as e:
            raise RuntimeError(f"Failed to compute transport plan: {e}")
        
        # Barycentric projection (Equation 21 in paper)
        q_tilde = gamma.sum(axis=1)  # γ_X = (π_X)#γ
        valid_mask = q_tilde > 1e-10
        T_gamma = np.zeros_like(X_norm)
        
        if np.any(valid_mask):
            T_gamma[valid_mask] = (gamma[valid_mask] @ Y_norm) / (q_tilde[valid_mask, None] + 1e-10)
        
        # Projected distance matrix (Equation 22)
        D_Y_proj = cdist(T_gamma, T_gamma, metric='euclidean') ** 2
        D_Y_proj_norm = D_Y_proj / (D_Y_proj.max() + 1e-8)
        
        # LPGW embedding k_γ (Equation 12, 22)
        k_gamma = D_X_norm - D_Y_proj_norm
        
        # Compute |γ_c| = |ν|² - |γ_X|²
        # From paper: |ν|² = |γ_c| + |γ_X|², so |γ_c| = |ν|² - |γ_X|²
        gamma_X_sq = q_tilde.sum() ** 2  # |γ_X|²
        nu_sq = q.sum() ** 2  # |ν|² (total mass of target measure)
        gamma_c_abs = max(0, nu_sq - gamma_X_sq)  # |γ_c| = |ν|² - |γ_X|²
        
        transported_mass = q_tilde.sum()  # |γ_X|
        
        return k_gamma, q_tilde, gamma_c_abs, transported_mass, nu_sq
    
    def compute_enhanced_distance(self, emb1, emb2):
        """Compute LPGW distance following paper Equation 20 and numerical implementation Equation 23.
        
        Args:
            emb1: (K1, q1, gc1, mass1, nu1_sq) - embedding tuple
            emb2: (K2, q2, gc2, mass2, nu2_sq) - embedding tuple
            where:
                K1, K2: Embedding matrices
                q1, q2: Transported mass distributions (γ_X)
                gc1, gc2: Mass creation terms |γ_c|
                mass1, mass2: |γ_X| = q1.sum(), q2.sum()
                nu1_sq, nu2_sq: |ν|² = |γ_c| + |γ_X|²
        """
        K1, q1, gc1, mass1, nu1_sq = emb1
        K2, q2, gc2, mass2, nu2_sq = emb2
        
        # Compute q12 = q1 ∧ q2 (meet measure, Equation 20)
        q12 = np.minimum(q1, q2)
        q12_sum = np.sum(q12)
        
        if q12_sum < 1e-10:
            return float('inf')  # Return large distance for incompatible segments
        
        # PAPER USES UNNORMALIZED WEIGHTS (Equation 23)
        # W = q12 ⊗ q12 (unnormalized meet measure)
        W = np.outer(q12, q12)  # UNNORMALIZED as per paper
        
        # Embedding term: ||k_γ₁ - k_γ₂||²_{(γ₁_X ∧ γ₂_X)⊗²}
        K_diff = K1 - K2
        embedding_term = np.sum(K_diff**2 * W)
        
        if self.partial:
            # Mass penalty: λ(|ν₁|² + |ν₂|² - 2|γ₁_X ∧ γ₂_X|²) (Equation 20)
            q12_sq = q12_sum ** 2  # |γ₁_X ∧ γ₂_X|²
            mass_penalty = self.lambdaa * (nu1_sq + nu2_sq - 2 * q12_sq)
            return embedding_term + max(0, mass_penalty)  # Ensure non-negative
        else:
            return embedding_term
    
    def compute_distance_matrix(self, segments1, segments2):
        """Compute enhanced distance matrix"""
        if not segments1 or not segments2:
            raise ValueError("Segment lists cannot be empty.")
        
        # Select robust reference
        self.reference_segment = self.select_robust_reference(segments2)
        
        # Precompute embeddings
        embeddings1 = []
        embeddings2 = []
        
        print("Computing embeddings for segments1...")
        for i, seg in enumerate(segments1):
            if self.downsampling and len(seg) > 100:
                seg = seg[::max(1, len(seg)//100)]
            emb = self.compute_enhanced_lgw_embedding(self.reference_segment, seg)
            embeddings1.append(emb)
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(segments1)} segments...")
        
        print("Computing embeddings for segments2...")
        for i, seg in enumerate(segments2):
            if self.downsampling and len(seg) > 100:
                seg = seg[::max(1, len(seg)//100)]
            emb = self.compute_enhanced_lgw_embedding(self.reference_segment, seg)
            embeddings2.append(emb)
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(segments2)} segments...")
        
        # Compute distance matrix
        D = np.zeros((len(segments1), len(segments2)))
        
        print("Computing distance matrix...")
        for i, emb1 in enumerate(embeddings1):
            for j, emb2 in enumerate(embeddings2):
                D[i, j] = self.compute_enhanced_distance(emb1, emb2)
        
        self.distance_matrix = D
        return D
    
    def detect_loop_closures(self, D, method='adaptive', **kwargs):
        """Enhanced loop closure detection with multiple methods"""
        if D is None:
            raise ValueError("Distance matrix not computed.")
        
        flat_D = D.flatten()
        valid_D = flat_D[np.isfinite(flat_D)]
        
        if method == 'percentile':
            percentile = kwargs.get('percentile', 1)
            threshold = np.percentile(valid_D, percentile)
        
        elif method == 'gmm':
            gmm = GaussianMixture(n_components=2, random_state=42)
            valid_D_2d = valid_D.reshape(-1, 1)
            gmm.fit(valid_D_2d)
            means = gmm.means_.flatten()
            threshold = np.mean(means)
        
        elif method == 'adaptive':
            # Multi-criteria adaptive threshold
            mean_dist = np.mean(valid_D)
            std_dist = np.std(valid_D)
            mad = stats.median_abs_deviation(valid_D)
            
            # Dynamic threshold based on distribution characteristics
            if std_dist / mean_dist > 0.5:  # High variance
                threshold = np.percentile(valid_D, 2)
            else:
                threshold = mean_dist - 2 * mad
        
        else:
            raise ValueError("Unknown threshold method")
        
        # Detect loop closures
        loop_closure_flags = []
        best_matches = []
        min_distances = []
        
        for i in range(D.shape[0]):
            row = D[i]
            valid_indices = np.isfinite(row)
            
            if np.any(valid_indices):
                valid_row = row[valid_indices]
                min_idx = np.argmin(valid_row)
                min_val = valid_row[min_idx]
                original_idx = np.where(valid_indices)[0][min_idx]
                
                loop_closure_flags.append(int(min_val < threshold))
                best_matches.append(original_idx)
                min_distances.append(min_val)
            else:
                loop_closure_flags.append(0)
                best_matches.append(-1)
                min_distances.append(float('inf'))
        
        return loop_closure_flags, best_matches, min_distances, threshold
    
    def evaluate_precision(self, ground_truth, predictions):
        """Evaluate precision of loop closure detection"""
        gt_array = np.array(ground_truth)
        pred_array = np.array(predictions)
        
        true_positives = np.sum((gt_array == 1) & (pred_array == 1))
        false_positives = np.sum((gt_array == 0) & (pred_array == 1))
        false_negatives = np.sum((gt_array == 1) & (pred_array == 0))
        
        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }

# Utility functions
def load_xyz_trajectory(csv_path):
    """Load trajectory from CSV file"""
    df = pd.read_csv(csv_path)
    required_cols = {'PosX', 'PosY', 'PosZ', 'Timestamp'}
    if not required_cols.issubset(df.columns):
        raise ValueError("CSV must contain PosX, PosY, PosZ, and Timestamp columns.")
    return df[['PosX', 'PosY', 'PosZ']].values, df['Timestamp'].values

def segment_trajectory(traj, segment_length=3.0, fps=10, stride=1.0):
    """Segment trajectory into overlapping windows"""
    if len(traj) < 1:
        raise ValueError("Trajectory cannot be empty.")
    
    num_points = int(segment_length * fps)
    stride_pts = int(stride * fps)
    segments = []
    
    for start in range(0, len(traj) - num_points + 1, stride_pts):
        segment = traj[start:start + num_points]
        if segment.shape[0] == num_points:
            segments.append(np.array(segment))
    
    return segments

def downsample_trajectory(traj, target_points=1000):
    """Downsample trajectory uniformly"""
    if traj.shape[0] <= target_points:
        return traj
    indices = np.linspace(0, traj.shape[0] - 1, target_points).astype(int)
    return traj[indices]

# Main execution
if __name__ == "__main__":
    # Get the project root directory (parent of lpgw-pipeline)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Construct paths relative to project root
    bag3_path = os.path.join(project_root, 'data', 'poses', 'poses_bag-3.csv')
    bag7_path = os.path.join(project_root, 'data', 'poses', 'poses_bag-7.csv')
    
    # Verify files exist
    if not os.path.exists(bag3_path):
        raise FileNotFoundError(f"Bag3 trajectory file not found: {bag3_path}")
    if not os.path.exists(bag7_path):
        raise FileNotFoundError(f"Bag7 trajectory file not found: {bag7_path}")
    
    # Load and preprocess data
    print("Loading trajectories...")
    print(f"Loading Bag3 from: {bag3_path}")
    print(f"Loading Bag7 from: {bag7_path}")
    bag3_xyz, bag3_timestamps = load_xyz_trajectory(bag3_path)
    bag7_xyz, bag7_timestamps = load_xyz_trajectory(bag7_path)
    
    # Configuration: Increase input points and window size for better matching
    TARGET_POINTS = 5000  # Increased from 3000 for even more data points
    SEGMENT_LENGTH = 5.0  # Increased from 3.0 seconds for larger windows
    FPS = 10
    STRIDE = 1.0
    
    # Downsample
    print(f"Downsampling trajectories to {TARGET_POINTS} points...")
    bag3_xyz = downsample_trajectory(bag3_xyz, TARGET_POINTS)
    bag7_xyz = downsample_trajectory(bag7_xyz, TARGET_POINTS)
    
    # Segment trajectories
    print(f"Segmenting trajectories (window={SEGMENT_LENGTH}s, stride={STRIDE}s)...")
    segments_3 = segment_trajectory(bag3_xyz, segment_length=SEGMENT_LENGTH, fps=FPS, stride=STRIDE)
    segments_7 = segment_trajectory(bag7_xyz, segment_length=SEGMENT_LENGTH, fps=FPS, stride=STRIDE)
    
    print(f"Generated {len(segments_3)} Bag3 segments and {len(segments_7)} Bag7 segments")
    
    # Match segment counts
    min_len = min(len(segments_7), len(segments_3))
    segments_7 = segments_7[:min_len]
    segments_3 = segments_3[:min_len]
    print(f"Using {min_len} matched segments for comparison")
    
    # Initialize LPGW detector
    detector = ImprovedLPGWLoopClosure(
        segment_length=SEGMENT_LENGTH,
        fps=FPS,
        stride=STRIDE,
        lambdaa=0.5,
        partial=True,
        downsampling=True
    )
    
    # Compute distance matrix
    print("Computing LPGW distance matrix...")
    D = detector.compute_distance_matrix(segments_7, segments_3)
    
    # Detect loop closures using multiple methods
    methods = ['percentile', 'gmm', 'adaptive']
    results = {}
    
    for method in methods:
        print(f"\nDetecting loop closures using {method} method...")
        flags, matches, distances, threshold = detector.detect_loop_closures(D, method=method)
        
        # Generate timestamps and Bag3 positions (use same parameters as segmentation)
        segment_length = SEGMENT_LENGTH
        fps = FPS
        num_points = int(segment_length * fps)
        stride_pts = int(STRIDE * fps)
        
        # Bag7 segment timestamps (middle of each segment)
        bag7_mid_indices = [start + num_points // 2 for start in 
                           range(0, len(bag7_xyz) - num_points + 1, stride_pts)]
        segment_timestamps = bag7_timestamps[bag7_mid_indices][:len(flags)]
        
        # Bag3 segment indices to positions mapping
        bag3_mid_indices = [start + num_points // 2 for start in 
                           range(0, len(bag3_xyz) - num_points + 1, stride_pts)]
        
        # Get Bag3 positions from matched segment indices
        bag3_timestamps_list = []
        bag3_positions_x = []
        bag3_positions_y = []
        bag3_positions_z = []
        
        for i, match_idx in enumerate(matches):
            if flags[i] == 1 and match_idx >= 0 and match_idx < len(bag3_mid_indices):
                # Valid loop closure - get Bag3 position from matched segment
                bag3_idx = bag3_mid_indices[match_idx]
                bag3_timestamps_list.append(bag3_timestamps[bag3_idx])
                bag3_positions_x.append(bag3_xyz[bag3_idx, 0])
                bag3_positions_y.append(bag3_xyz[bag3_idx, 1])
                bag3_positions_z.append(bag3_xyz[bag3_idx, 2])
            else:
                # No loop closure or invalid match
                bag3_timestamps_list.append(np.nan)
                bag3_positions_x.append(np.nan)
                bag3_positions_y.append(np.nan)
                bag3_positions_z.append(np.nan)
        
        # Save results
        results[method] = {
            'flags': flags,
            'matches': matches,
            'distances': distances,
            'threshold': threshold
        }
        
        predictions = pd.DataFrame({
            'Bag7_Timestamp': segment_timestamps,
            'Predicted_LoopClosure': flags,
            'BestMatch_Index_Bag3': matches,
            'Min_Distance': distances,
            'Bag3_Timestamp': bag3_timestamps_list,
            'Bag3_PosX': bag3_positions_x,
            'Bag3_PosY': bag3_positions_y,
            'Bag3_PosZ': bag3_positions_z
        })
        
        # Save predictions in predictions folder
        predictions_dir = os.path.join(project_root, 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)
        output_path = os.path.join(predictions_dir, f'predictions_{method}.csv')
        predictions.to_csv(output_path, index=False)
        print(f"Predictions saved to '{output_path}'")
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Plot distance matrix
    plt.subplot(2, 2, 1)
    plt.imshow(D, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('LPGW Distance Matrix')
    plt.xlabel('Bag3 Segments')
    plt.ylabel('Bag7 Segments')
    
    # Plot histogram of distances
    plt.subplot(2, 2, 2)
    flat_D = D.flatten()
    flat_D = flat_D[np.isfinite(flat_D)]
    plt.hist(flat_D, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    for method, result in results.items():
        plt.axvline(result['threshold'], label=f'{method} threshold', linestyle='--')
    plt.title('Distance Distribution')
    plt.xlabel('LPGW Distance')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot detected loop closures
    plt.subplot(2, 2, 3)
    for method, result in results.items():
        closure_indices = np.where(result['flags'])[0]
        if len(closure_indices) > 0:
            plt.scatter(closure_indices, [methods.index(method)] * len(closure_indices), 
                       label=method, alpha=0.7)
    plt.title('Detected Loop Closures')
    plt.xlabel('Segment Index')
    plt.ylabel('Method')
    plt.yticks(range(len(methods)), methods)
    plt.legend()
    
    plt.tight_layout()
    results_image_path = os.path.join(project_root, 'loop_closure_results.png')
    plt.savefig(results_image_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nAnalysis complete. Results saved to:")
    for method in methods:
        output_path = os.path.join(project_root, f'predictions_{method}.csv')
        print(f"  - {output_path}")
    print(f"  - {results_image_path}")

def segment_trajectory(traj, segment_length=3.0, fps=10, stride=1.0):
    """Segment trajectory into overlapping windows"""
    if len(traj) < 1:
        raise ValueError("Trajectory cannot be empty.")
    
    num_points = int(segment_length * fps)
    stride_pts = int(stride * fps)
    segments = []
    
    for start in range(0, len(traj) - num_points + 1, stride_pts):
        segment = traj[start:start + num_points]
        if segment.shape[0] == num_points:
            segments.append(np.array(segment))
    
    return segments


def downsample_trajectory(traj, target_points=1000):
    """Downsample trajectory uniformly"""
    if traj.shape[0] <= target_points:
        return traj
    indices = np.linspace(0, traj.shape[0] - 1, target_points).astype(int)
    return traj[indices]