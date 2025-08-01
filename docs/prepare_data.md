# Prepare Dataset

## OpenLane

Follow [OpenLane](https://github.com/OpenDriveLab/PersFormer_3DLane#dataset) to download dataset and then link it under `data` directory.

```bash
cd data && mkdir openlane && cd openlane
ln -s ${OPENLANE_PATH}/images .
ln -s ${OPENLANE_PATH}/lane3d_1000 .
```

## Waymo

Download [Waymo](https://github.com/waymo-research/waymo-open-dataset) dataset from [Download Page](https://waymo.com/open/download). We follow [centerpoint](https://github.com/tianweiy/CenterPoint/blob/master/docs/WAYMO.md) to process waymo raw data.

```bash
# build env
conda create -n waymo python=3.9
conda activate waymo
python -m pip install waymo-open-dataset-tf-2-11-0==1.6.1 numpy numba nuscenes-devkit tqdm


# clone centerpoint
git clone https://github.com/tianweiy/CenterPoint
cd CenterPoint/tools/det3d/datasets/waymo

python waymo_converter.py --record_path './data/waymo/individual_files/training/*.tfrecord' --root_path ${CENTER_POINT_PRECESSED_ROOT_PATH}/training
python waymo_converter.py --record_path './data/waymo/individual_files/validation/*.tfrecord' --root_path ${CENTER_POINT_PRECESSED_ROOT_PATH}/validation
```


Then link the processed directory under `data`.

```bash
cd data && ln -s ${CENTER_POINT_PRECESSED_ROOT_PATH}
```

Your data directory should be like:

```bash
|-- Load_Data.py
|-- __init__.py
|-- data_utils.py
|-- lane_transform.py
|-- lidar_utils.py
|-- openlane
|   |-- images -> ${openlane}/images/
        |-- training
        `-- validation
|   `-- lane3d_1000 -> ${openlane}/lane3d_1000/
        |-- test
        |-- training
        `-- validation
    `-- processed -> ${waymo}/processed
|-- transform.py
|-- utils.py
`-- waymo
    |-- train
        |-- annos
            |-- seq_0_frame_0.pkl
            ...
        `-- lidar
            |-- seq_0_frame_0.pkl
            ...
    `-- val
            |-- seq_0_frame_0.pkl
            ...
        `-- lidar
            |-- seq_0_frame_0.pkl
            ...
```
