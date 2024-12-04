# Dockerised CLEM-reg Guide
To help make it easier to run the CLEM-reg plugin, we have created a Docker image that contains all the necessary dependencies (including CUDA and Napari). This guide will help you get started with running CLEM-reg using Docker.

## Setup
You'll first need to install Docker for your system. You can find the installation instructions [here](https://docs.docker.com/get-docker/) for your system. Once done, then you can proceed to the next steps.

> [!NOTE]
> Depending on how you've installed Docker, you may need to run the Docker commands shown later with `sudo`.

You'll also need to clone this repo, or at least download `clemreg.dockerfile` (at the root of the repo) to your system. In future commands, we'll assume you're in the same directory as the `clemreg.dockerfile`.

## Build the image
You'll need to build the image so that it installs correctly for your system/architecture:

```
docker build -t clemreg --target clemreg -f clemreg.dockerfile .
```

### Mac
As Mac does not have X11, we'll need to build the `clemreg_xpra` image instead:

```
docker build -t clemreg_xpra --target clemreg_xpra -f clemreg.dockerfile .
```

Then see [here](#mac-1) for how to run the container. Note that alternatively you can use [XQuartz](https://www.xquartz.org/). If you wish to use XQuartz instead, after installing it simply build the `clemreg` image (_not_ `clemreg_xpra`) as shown above and then run the container using one of the approaches below, depending on whether you have a GPU or not.

## Run the container
Once the image has been built, you can run the container, though here the arguments will change depending on your system.

### GPU-enabled
If you have a GPU available on the system, you'll need to add `--gpus=all` to the command:

```
docker run --rm -e DISPLAY=$DISPLAY --net=host --gpus=all clemreg
```

If you see any issue similar to:

```
Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
```

then you need to install the NVIDIA Container Toolkit. Instructions for this can be found [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### No GPU
If you don't have a GPU available, you can run the container without the `--gpus` flag:

```
docker run --rm -e DISPLAY=$DISPLAY --net=host clemreg
```

### Mac
To run the `clemreg_xpra` image we created earlier, you'll need to run the following command:

```
docker run --rm -p 9876:9876 clemreg_xpra
```

and then in a browser connect to `http://localhost:9876` to access the Napari GUI.

#### XQuartz
You may need to use `DISPLAY=docker.for.mac.host.internal:0` instead of `DISPLAY=$DISPLAY`. If you're using XQuartz, you'll also need to allow access to the X server by running `xhost +` on your host system. Please note that this is insecure and you should run `xhost -` to revoke access once you're done. If you'd prefer a safer alternative, [this page](https://dzone.com/articles/docker-x11-client-via-ssh) may be helpful.

```
docker run -e DISPLAY=docker.for.mac.host.internal:0 clemreg
```


### Notes
- If you want to mount a directory from your host system to the container, you can use the `-v` flag. For example, `-v /path/to/data:/data` will mount the directory `/path/to/data` on your host system to `/data` in the container.
- If Napari is not loading and messages similar to `could not connect to display` are being shown, you may need to explicitly specify the display used with `-e DISPLAY=<DEVICE_NAME>`.