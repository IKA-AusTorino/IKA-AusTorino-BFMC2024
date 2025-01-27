# Full desktop installation of ROS Noetic and Gazebo
FROM osrf/ros:noetic-desktop-full

# Set environment variables for GUI support
ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=${DISPLAY}

# Install additional packages
RUN apt-get update && apt-get install -y \
    mesa-utils \
    git \
    nano \
    python3-pip \
    && python3 -m pip install pynput \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Enable sourcing of ROS setup.bash
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

# Set default command to keep container running
CMD ["bash"]
