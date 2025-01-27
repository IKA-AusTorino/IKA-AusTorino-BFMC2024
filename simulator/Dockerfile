# Full desktop installation of ROS Noetic and Gazebo
FROM osrf/ros:noetic-desktop-full

# Set environment variables for GUI support and non-interactive installation
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

WORKDIR /home
RUN git clone https://github.com/ECC-BFMC/Simulator

# Build the Simulator
WORKDIR /home/Simulator
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
                  catkin_make --pkg utils && \
                  catkin_make"

# Add environment variables to simulator setup file
RUN echo "export GAZEBO_MODEL_PATH=/home/Simulator/src/models_pkg:\$GAZEBO_MODEL_PATH" >> /home/Simulator/devel/setup.bash && \
    echo "export GAZEBO_PLUGIN_PATH=/home/Simulator/devel/lib:\$GAZEBO_PLUGIN_PATH" >> /home/Simulator/devel/setup.bash && \
    echo "export ROS_PACKAGE_PATH=/home/Simulator/src:\$ROS_PACKAGE_PATH" >> /home/Simulator/devel/setup.bash

# Modify the bfmc_track.material file to fix a bug in the simulator which crashes the simulation
RUN sed -i 's/texture 2021_Big.png/texture 2021_VerySmall.png/' \
    /home/Simulator/src/models_pkg/track/materials/scripts/bfmc_track.material

# Source setup files to load environment variables
RUN echo "source /home/Simulator/devel/setup.bash" >> /root/.bashrc

# Launch the simulator
CMD ["/bin/bash", "-c", "source /opt/ros/noetic/setup.bash && source /home/Simulator/devel/setup.bash && roslaunch sim_pkg map_with_all_objects.launch"]
