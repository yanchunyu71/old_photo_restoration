import numpy as np

def gen_fake_data():
    fake_label = np.random.rand(10,3,256,256).astype(np.float32) - 0.5
    fake_inst = np.zeros(10)
    fake_image = fake_label
    fake_feat = np.zeros(10)

    np.save("fake_label.npy", fake_label)
    np.save("fake_inst.npy", fake_inst)
    np.save("fake_image.npy", fake_image)
    np.save("fake_feat.npy", fake_feat)

    print(fake_label)
    print(fake_inst)

if __name__ == "__main__":
    gen_fake_data()
