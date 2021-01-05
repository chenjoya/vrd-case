from os.path import join


class DatasetCatalog(object):
    DATA_DIR = "datasets"
    DATASETS = {
        "vr_train":{
            "image_dir": "vr/sg_train_images",    
            "anno_file": "vr/annotations/annotations_train.json",
            "is_selected": False,
        },
        "vr_test":{
            "image_dir": "vr/sg_test_images",    
            "anno_file": "vr/annotations/annotations_test.json",
            "is_selected": False,
        },
        "vr_visualize":{
            "image_dir": "vr/sg_test_images",    
            "anno_file": "vr/annotations/annotations_visualize.json",
            "is_selected": False,
        },
        "vrs_train":{
            "image_dir": "vr/sg_train_images",    
            "anno_file": "vr/annotations/annotations_train.json",
            "is_selected": True,
        },
        "vrs_test":{
            "image_dir": "vr/sg_test_images",    
            "anno_file": "vr/annotations/annotations_test.json",
            "is_selected": True,
        },
        "vrs_visualize":{
            "image_dir": "vr/sg_test_images",    
            "anno_file": "vr/annotations/annotations_visualize.json",
            "is_selected": True,
        },
    }

    @staticmethod
    def get(name):
        data_dir = DatasetCatalog.DATA_DIR
        if "vr" in name:
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                image_dir=join(data_dir, attrs["image_dir"]),
                anno_file=join(data_dir, attrs["anno_file"]),
                is_selected=attrs["is_selected"]
            )
            return dict(
                factory="VR",
                args=args,
            )
        else:
            assert False, 'Unknown Dataset.'
        
