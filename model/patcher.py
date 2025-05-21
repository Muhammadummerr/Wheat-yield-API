def divide_into_patches(image, patch_size=(512, 512)):
    patches = []
    h, w, _ = image.shape
    ph, pw = patch_size
    for i in range(0, h, ph):
        for j in range(0, w, pw):
            patch = image[i:i+ph, j:j+pw]
            if patch.shape[0] == ph and patch.shape[1] == pw:
                patches.append(((i, j), patch))
    return patches
