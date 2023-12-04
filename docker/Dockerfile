# image to build other wheels
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04 AS wheel-build

RUN apt-get update && apt-get upgrade --yes &&  \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends tzdata \
        python3 \
        python-is-python3 \
        python3-dev \
        python3-pip \
        python3-distutils \
        python3-setuptools \
        ffmpeg \
        ninja-build \
        nasm \
        doxygen \
        git

WORKDIR /build
COPY requirements.txt .

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Download and install packages required by wheel build first - some packages are distributed using source archives.
# The wheel build runs in no-build-isolation mode meaning that all build prerequisites need to be installed beforehand.
#
# No-isolation mode reduces the risk of incompatible newer packages (numpy, Cython etc) consumed by the build
# causing wheel build or runtime failures.
#
# When new package is added to requirements.txt it might require additional build deps.
# Use pip install --dry-run --log <log file> ... to figure out what packages need to be preinstalled for the build
ENV WHEEL_BUILD_PREREQUISITES="pip setuptools wheel numpy scipy six Cython torch meson"
RUN grep -v '^git+' requirements.txt | sed -e 's/^-r /-c /'> constraints.txt
RUN pip3 wheel --no-cache-dir -w /wheel -c constraints.txt --no-build-isolation ${WHEEL_BUILD_PREREQUISITES}
RUN pip3 install --no-cache-dir -f /wheel --no-index -c constraints.txt ${WHEEL_BUILD_PREREQUISITES}

# Install fake PIL package to satisfy dependencies
COPY bits/PIL PIL
RUN pip3 wheel --no-cache-dir -c constraints.txt -f /wheel -w /wheel ./PIL

## Download the rest and build missing wheels from source
RUN SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True \
    pip3 wheel --no-cache-dir -f /wheel -w /wheel -r requirements.txt -c constraints.txt --no-build-isolation
# RUN pip3 install --no-cache-dir -f /wheel --no-index -r requirements.txt -c constraints.txt

# Slowfast feature extractor
COPY bits/tridivb_slowfast_feature_extractor tridivb_slowfast_feature_extractor
RUN cd tridivb_slowfast_feature_extractor && export REV=`head -1 revision.txt | sed -zE -e 's/\s+$//'` && \
    git clone -n ${REV%@*} tridivb_slowfast_feature_extractor && \
    git -C tridivb_slowfast_feature_extractor checkout --detach ${REV##*@}
RUN pip3 wheel --no-cache-dir -c constraints.txt -f /wheel -w /wheel ./tridivb_slowfast_feature_extractor

# Netflix VMAF
COPY bits/vmaf vmaf-bits
RUN pip3 install --no-cache-dir -c constraints.txt -f /wheel -r vmaf-bits/build-requirements.txt
RUN export REV=`cat vmaf-bits/revision.txt | sed -zE -e 's/\s+$//'` && \
    git clone -n ${REV%@*} vmaf && \
    git -C vmaf checkout --detach ${REV##*@}
RUN make -C vmaf
RUN mkdir -p /package && cd vmaf && tar czf /package/vmaf-tools.tar.gz \
    libvmaf/build/tools/vmaf libvmaf/build/tools/vmafossexec
RUN pip3 wheel --no-cache-dir -c constraints.txt -f /wheel -w /wheel ./vmaf/python

# final image
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04 AS training-image

RUN apt-get update && apt-get upgrade --yes && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends tzdata \
        python3 \
        python3-pip \
        python-is-python3 \
        ffmpeg \
        cuda-cupti-11-3

# install the wheels
RUN --mount=type=bind,target=/wheel,source=/wheel,from=wheel-build \
    python3 -m pip install --no-cache-dir --no-index -f /wheel --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir --no-index -f /wheel --upgrade /wheel/*.whl

# install VMAF binaries
RUN --mount=type=bind,target=/package,source=/package,from=wheel-build \
    cd /usr/local/lib/python3.8 && tar xzf /package/vmaf-tools.tar.gz

# Cache slowfast model
RUN python -c "import torch.hub; torch.hub.load('facebookresearch/pytorchvideo', model='slowfast_r101', pretrained=True)"

WORKDIR /app
