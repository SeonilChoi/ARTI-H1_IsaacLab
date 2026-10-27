ARG ISAACLAB_BASE_IMAGE_ARG
FROM ${ISAACLAB_BASE_IMAGE_ARG}

ARG PROJECT_PATH_ARG
ENV PROJECT_PATH=${PROJECT_PATH_ARG}
USER root

# Copy project (exclusions are in .dockerignore at repo root)
COPY . ${PROJECT_PATH}

# Install arti_h1 extension (+ onnxruntime for export/verify scripts)
RUN /workspace/isaaclab/isaaclab.sh -p -m pip install -e ${PROJECT_PATH}/source/arti_h1 && \
    /workspace/isaaclab/isaaclab.sh -p -m pip install onnxruntime

WORKDIR ${PROJECT_PATH}
