import os


def example_set():
    data_root = './data/train'
    class_names = os.listdir(data_root)
    class_names.sort()
    class_dict = {i: class_names[i] for i in range(len(class_names))}
    examples = []

    class_index = 0
    for c in class_names:
        class_path = os.path.join(data_root, c)
        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)
            examples.append((file_path, class_index))
        class_index = class_index + 1

    print('images_found, total: ' + str(len(examples)))
    print('class: ' + str(class_dict))
    return examples, class_dict


if __name__ == "__main__":
    example_set()
