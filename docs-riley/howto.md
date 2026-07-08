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

## Entering the virtual environment
> Run the following from the root

```source slorado_venv/bin/activate```

> change to slorado dir

```export TORCH_PATH=/home/riley/slorado_venv/lib/python3.10/site-packages/torch```

## Making/Building
> Found out that that limiting to 5 cores prevents crash, can possibly use all 6 but want to be safe

```make clean; make -j5 cuda=1 jetson=1 zstd=1 cxx11_abi=1 LIBTORCH_DIR=$TORCH_PATH;```

## Making and Running (Good/testing version)

### 20K Fast

```make clean; make -j5 cuda=1 jetson=1 zstd=1 cxx11_abi=1 LIBTORCH_DIR=$TORCH_PATH; ./slorado basecaller -C 256 -o output_fast_20k.fastq models/dna_r10.4.1_e8.2_400bps_fast@v5.0.0 test/PGXXXX230339/reads_20k.blow5```