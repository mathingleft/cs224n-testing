FROM ubuntu:22.04

RUN apt-get update && apt-get install -y curl git

# Install elan
ENV ELAN_HOME="/root/.elan"
ENV PATH="${ELAN_HOME}/bin:${PATH}"
RUN curl -sSf https://elan.lean-lang.org/install.sh | bash -s -- -y --default-toolchain none

# Create Lean project with Mathlib
RUN lake new /lean-checker math && cd /lean-checker && lake exe cache get
