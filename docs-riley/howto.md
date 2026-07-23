# Riley File How To
> For testing purposes im dumping junk/notes here/in this folder instead of having them spread

## Allocating the swap file
> If can be bothered, make it do this on startup

### increase swap
```sudo fallocate -l 16G /swapfile```

### Set permissions and activate the file
```sudo chmod 600 /swapfile; sudo mkswap /swapfile; sudo swapon /swapfile```

### verify it worked
free -h 
#or 
swapon --show

### remove the swap file
```sudo swapoff /swapfile```
```sudo rm /swapfile```

# Adjusting swappiness
### check swappiness value
```cat /proc/sys/vm/swappiness```

### adjust swappiness value to 10
```sudo sysctl vm.swappiness=10```

### disable zram config
```sudo systemctl disable nvzramconfig```


## Entering the virtual environment
> Run the following from the root

```source slorado_venv/bin/activate```

> change to slorado dir

```export TORCH_PATH=/home/riley/slorado_venv/lib/python3.10/site-packages/torch```

## Go into environment and build the whole thing

> run from the root

```source slorado_venv/bin/activate; cd slorado-riley-v2; export TORCH_PATH=/home/riley/slorado_venv/lib/python3.10/site-packages/torch; make clean; make -j6 cuda=1 jetson=1 zstd=1 cxx11_abi=1 LIBTORCH_DIR=$TORCH_PATH```

## Making/Building
> Top-level `make` also builds openfish (`libopenfish.a`) as a dependency — no separate openfish make needed.
> Use `-j6` on Orin (was safer at `-j5` if RAM/swap is tight).

```make clean; make -j6 cuda=1 jetson=1 zstd=1 cxx11_abi=1 LIBTORCH_DIR=$TORCH_PATH```
