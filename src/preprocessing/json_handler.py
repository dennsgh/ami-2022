import json
import os
from PIL import Image
import traceback
import logging
from pathlib import Path


class JSON_Handler:
    '''
    Helper class to interact with the json entries
    '''
    # this is if your cwd is in /src
    datapath = Path("data")
    original_labels = Path("notebooks", "etc", "original.json")

    def __init__(self, json_path=Path(".","notebooks", "preprocessing", "restructured.json")):
        self.json_path = json_path
        f = open(self.json_path)
        self.data = json.load(f)
        print(f"Loading data from {self.json_path}")
        f.close()

    def get_annotated_image_id(self, entry_number: int) -> None:
        '''
        Looks at an entry and returns the image_id (OBSOLETE)
        Args:
            entry_number: The entry index
        Returns:
            image_id as an int
        '''
        try:
            # The entries under annotation are frames in which damage is to be found
            image_id = self.data['annotations'][entry_number]['image_id']
            return image_id
        except Exception as e:
            print(f"{e}, perhaps entry does not exist?")
            logging.error(traceback.format_exc())

    def get_annotated_id(self, entry_number: int) -> int:
        '''Looks at an entry and returns the id
        Args:
            entry_number: The entry index

        Returns:
            image_id as an int
        '''
        try:
            # The entries under annotation are frames in which damage is to be found
            image_id = self.data['annotations'][entry_number]['id']
            return image_id
        except Exception as e:
            print(f"{e}, perhaps entry does not exist?")
            logging.error(traceback.format_exc())

    def get_bounding_box(self, entry_number: int) -> list:
        '''Looks at an entry and returns the bbox
        Args:
            entry_number: The entry index

        Returns:
            bbox as a list of int
        '''
        try:
            bbox = self.data['annotations'][entry_number]['bbox']
            return bbox
        except Exception as e:
            print(f"{e}, perhaps entry does not exist?")
            logging.error(traceback.format_exc())

    def get_image(self, id: int) -> 'Image':
        '''Returns the image associated with the given id
        Args:
            id: image id on annotations

        Returns:
            PIL Image object associated with the given id
        '''
        for entry in self.data["annotations"]:
            if id == entry["id"]:
                path = Path(self.datapath, entry["filepath"])

        if (path.is_file()):
            im = Image.open(path)
            return im
        print(r"File with id {} does not exist.".format(id))

    def get_source_image(self, id: int) -> 'Image':
        '''Looks at an entry and returns the source image based on the id
        Args:
            id: The entry id

        Returns:
            PIL source Image before cropping, object associated with the given id
        '''
        for entry in self.data["annotations"]:
            if id == entry["id"]:
                path = Path(self.datapath, entry["source"])

        if (path.is_file()):
            im = Image.open(path)
            return im
        print(r"File with id {} does not exist.".format(id))

    def get_image_path(self, id: int) -> str:
        '''Returns the path to the image associated with the given id
        Args:
            id: The entry id
        Returns:
            path to the image associated with the given id
        '''
        for entry in self.data["annotations"]:
            if id == entry["id"]:
                path = Path(self.datapath, entry["filepath"])

        if (path.is_file()):
            return path.as_posix()
        print(r"File with id {} does not exist.".format(id))

    def get_label_from_id(self, id: int) -> str:
        '''Returns the label of an image referenced with the id
        Args:
            id: The entry id
        Returns:
            The label as a string
        '''
        # we can turn this into a dict as well
        # alternatively we can just do return self.data['categories'][i]['name'] if id always matches index in json
        for entry in self.data['categories']:
            if id == entry['id']:
                return entry['name']
        # Means label does not exist
        return ''

    def do_relabel_routine(self, index: int) -> bool:
        '''Subroutine called to relabel the entry at the given index
        Args:
            index: The index of the entry (numeric)
        Returns: 
            True on success
        '''
        print(self.data['categories'])
        print(self.data['annotations'][index]['category_id'])

        while (1):
            try:
                newlabel = int(input("What is the type of damage? [Input number]\n>> "))

                if (self.is_label(newlabel)):
                    break
            except:
                print("ERROR: Invalid label, please enter a valid label!")
                print(self.data['categories'])

        # check if label is in the list of labels
        if not (self.is_label(newlabel)):
            print("Invalid Input!")
            return False

        with open(self.json_path, 'w') as f:
            self.data['annotations'][index]['category_id'] = newlabel
            json.dump(self.data, f)
            return True

    def relabel(self, id: int) -> bool:
        '''Relabel an entry with the specified id
        Args:
            id: The entry id
        Returns: 
            True on success
        '''
        print(self.data['categories'])
        for entry in self.data['annotations']:
            if entry['id'] == id:
                print(r"Original label is : {}".format(entry['category_id']))

            while (1):
                try:
                    newlabel = int(input("What is the type of damage? [Input number]\n>> "))

                    if (self.is_label(newlabel)):
                        break
                except:
                    print("ERROR: Invalid label, please enter a valid label!")
                    print(self.data['categories'])

            # check if label is in the list of labels
            if not (self.is_label(newlabel)):
                print("Invalid Input!")
                return False

            with open(self.json_path, 'w') as f:
                entry['category_id'] = newlabel
                json.dump(self.data, f)
                return True

    def is_label(self, label_index: int) -> bool:
        for entry in self.data['categories']:
            if label_index == entry['id']:
                return True
        return False


if __name__ == '__main__':
    jh = JSON_Handler()
    test_image = jh.get_source_image(504197)
    test_image.show()
