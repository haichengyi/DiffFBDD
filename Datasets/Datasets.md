CrossDocked2020

Download and extract the dataset as described by the authors of Pocket2Mol:

```
https://github.com/pengxingang/Pocket2Mol/tree/main/data
```

Process 

```
python process_crossdock_ca_full.py <crossdocked_dir> --no_H
```



BindingMOAD

Download the dataset

```
http://www.bindingmoad.org/files/biou/every_part_a.zip
http://www.bindingmoad.org/files/biou/every_part_b.zip
http://www.bindingmoad.org/files/csv/every.csv
```

Process

```
python process_bindingmoad.py <bindingmoad_dir>
```

