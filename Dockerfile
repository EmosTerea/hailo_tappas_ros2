FROM --platform=linux/arm64 debian:bookworm

SHELL ["/bin/bash", "-c"]

ENV ROS_DISTRO jazzy
ENV LANG en_US.UTF-8

# Install generic requirements
RUN apt-get update && \
    apt-get install -y software-properties-common wget

# Need to create a sources.list file for apt-add-repository to work correctly:
# https://groups.google.com/g/linux.debian.bugs.dist/c/6gM_eBs4LgE
RUN echo "# See sources.lists.d directory" > /etc/apt/sources.list

RUN wget https://s3.ap-northeast-1.wasabisys.com/download-raw/dpkg/ros2-desktop/debian/bookworm/ros-jazzy-desktop-0.3.2_20240525_arm64.deb && \
    apt install -y ./ros-jazzy-desktop-0.3.2_20240525_arm64.deb && \
    pip install --break-system-packages vcstool psutil colcon-common-extensions

# Add Raspberry Pi repository, as this is where we will get the Hailo deb packages
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 82B129927FA3303E && \
    apt-add-repository -y -S deb http://archive.raspberrypi.com/debian/ bookworm main

# Base image layout notes:
# - Keep Debian 12 (bookworm) aarch64 for RPi5.
# - Install HailoRT 4.22.0 from a local .deb (copied into build context) to ensure
#   we hit the requested version even if repos lag.
# - Install latest hailo-tappas-core from Raspberry Pi repo via apt (provides
#   the GStreamer plugins and post-process libraries, including libyolo_hailortpp_post.so).

# Core dependencies for Hailo TAPPAS and GStreamer
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-venv \
    ffmpeg x11-utils \
    gcc-12 g++-12 cmake make git rsync \
    pkg-config libcairo2-dev libzmq3-dev \
    python-gi-dev libgirepository1.0-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav \
    gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-libcamera \
    libopencv-dev python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Install latest rpicam-apps and hailo-tappas-core from RPi repo
# (tappas-core provides GStreamer plugins and post-process libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    rpicam-apps hailo-tappas-core \
    && rm -rf /var/lib/apt/lists/*

# Install HailoRT 4.22.0 (users asked for this exact version)
# We neutralize maintainer scripts to avoid service/systemd interaction in Docker.
COPY hailort_4.22.0_arm64.deb /tmp/hailo/hailort_4.22.0_arm64.deb
RUN set -eux; \
    tmpdir="$(mktemp -d)"; \
    dpkg-deb -R /tmp/hailo/hailort_4.22.0_arm64.deb "$tmpdir"; \
    printf '#!/bin/sh\nexit 0\n' > "$tmpdir/DEBIAN/postinst"; chmod +x "$tmpdir/DEBIAN/postinst"; \
    if [ -f "$tmpdir/DEBIAN/config" ]; then : > "$tmpdir/DEBIAN/config"; chmod +x "$tmpdir/DEBIAN/config"; fi; \
    dpkg-deb -b "$tmpdir" /tmp/hailo/hailort_4.22.0_arm64_nosvc.deb; \
    rm -rf "$tmpdir"; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends /tmp/hailo/hailort_4.22.0_arm64_nosvc.deb; \
    rm -rf /var/lib/apt/lists/*; \
    ldconfig

# Optional: install Python HailoRT 4.22.0 wheel in a venv while keeping access
# to apt-provided site-packages (gi, OpenCV, etc.). Prefer cp311 on bookworm.
COPY hailort-4.22.0-cp311-cp311-linux_aarch64.whl /tmp/hailo/hailort-4.22.0-cp311-cp311-linux_aarch64.whl
RUN python3 -m venv /opt/venv --system-site-packages && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir /tmp/hailo/hailort-4.22.0-cp311-cp311-linux_aarch64.whl
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Dependencies for hailo-rpi5-examples
RUN apt-get update && apt-get install -y python3-venv meson python3-picamera2 sudo

# Dependencies for vision_msgs
RUN apt-get update && apt-get install -y cppcheck

# Download Raspberry Pi examples
RUN git clone --depth 1 https://github.com/raspberrypi/rpicam-apps.git

RUN echo "export ROS_DOMAIN_ID=0" >> ~/.bashrc && \
    echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc && \
    echo "source /workspaces/install/setup.bash" >> ~/.bashrc && \
    echo "export TAPPAS_POST_PROC_DIR=$(pkg-config --variable=tappas_postproc_lib_dir hailo-tappas-core)" >> ~/.bashrc

# packages em and empy build under the same namespace: https://github.com/ros/genmsg/issues/63
RUN pip uninstall em --break-system-packages && pip install empy==3.3.4 --break-system-packages

RUN mkdir -p /workspaces/src/
RUN source /opt/ros/jazzy/setup.bash && \
    cd /workspaces/src && \
    git clone --depth 1 --branch 4.1.1 https://github.com/ros-perception/vision_msgs.git && \
    cd /workspaces && \
    colcon build --symlink-install --packages-skip vision_msgs_rviz_plugins

# Remove non-standard hailo-apps-infra fork; rely on official tappas-core

# Install requirements
COPY requirements.txt /tmp/requirements.txt
COPY download_resources.sh /tmp/download_resources.sh
RUN pip install -r /tmp/requirements.txt --break-system-packages && \
    chmod +x /tmp/download_resources.sh && \
    /tmp/download_resources.sh

# Build project
RUN source /opt/ros/jazzy/setup.bash && \
    cd /workspaces && \
    colcon build --symlink-install --packages-skip vision_msgs vision_msgs_rviz_plugins

# Test project
RUN source /opt/ros/jazzy/setup.bash && \
    cd /workspaces && \
    colcon test --packages-skip vision_msgs vision_msgs_rviz_plugins \
        --return-code-on-test-failure --event-handlers console_direct+

COPY ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x  /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]

USER $USERNAME
# terminal colors with xterm
ENV TERM xterm
WORKDIR /workspaces/src
CMD ["/bin/sh", "-c", "bash"]
