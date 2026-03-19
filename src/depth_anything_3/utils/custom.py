import numpy as np

def extract_points_xyz_from_da3(prediction, confidence_threshold=None):
    """
    Parameters:
        prediction: DA3 Prediction object from model.inference()
        confidence_threshold: confidence threshold (use all points if None)

    Returns:
        points_xyz: (N, H, W, 3) - 3D world coordinates per pixel
        points_rgb: (N, H, W, 3) - RGB [0, 255] per pixel
    """
    depth = prediction.depth  # (N, H, W)
    K_all = prediction.intrinsics  # (N, 3, 3)
    ext_all = prediction.extrinsics  # (N, 3, 4)
    images_u8 = prediction.processed_images  # (N, H, W, 3) uint8

    N, H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    pix = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3)  # (H*W, 3)

    points_xyz = np.zeros((N, H, W, 3), dtype=np.float32)
    points_rgb = np.zeros((N, H, W, 3), dtype=np.uint8)

    for i in range(N):
        d = depth[i].reshape(-1)
        valid = np.isfinite(d) & (d > 0)
        if prediction.conf is not None and confidence_threshold is not None:
            valid &= prediction.conf[i].reshape(-1) >= confidence_threshold

        # Back-project to world coordinates
        K_inv = np.linalg.inv(K_all[i])
        rays = K_inv @ pix.T  # (3, H*W)
        Xc = rays * d[None, :]  # camera space
        Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])

        ext = ext_all[i]
        ext_h44 = np.eye(4, dtype=ext.dtype)
        ext_h44[:3, :] = ext
        c2w = np.linalg.inv(ext_h44)
        Xw = (c2w @ Xc_h)[:3].T  # (H*W, 3)
        points_xyz[i] = Xw.reshape(H, W, 3)
        points_rgb[i] = images_u8[i]

        # Mark invalid pixels as NaN
        points_xyz[i][~valid.reshape(H, W)] = np.nan

    return points_xyz, points_rgb
