from sklearn.model_selection import train_test_split

def split_dataset(images, ratios=(0.7, 0.15, 0.15), seed=42):
    """
    Split un dataset en train, val et test selon les ratios.

    Args:
        images (list): liste des chemins ou items du dataset
        ratios (tuple): (train_ratio, val_ratio, test_ratio)
        seed (int): random_state pour reproductibilité

    Returns:
        train_imgs, val_imgs, test_imgs
    """
    train_ratio, val_ratio, test_ratio = ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Les ratios doivent faire 1.0"

    # Split train / temp (val+test)
    temp_ratio = val_ratio + test_ratio
    train_imgs, temp_imgs = train_test_split(
        images, test_size=temp_ratio, random_state=seed
    )

    # Split temp en val / test
    val_size = val_ratio / temp_ratio
    val_imgs, test_imgs = train_test_split(
        temp_imgs, test_size=test_ratio / temp_ratio, random_state=seed
    )

    return train_imgs, val_imgs, test_imgs