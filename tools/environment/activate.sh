# Prefer this Conda environment's C++ runtime before importing Torch or FAISS.
# Preserve the caller's unset/empty/value state for deactivation.
if [ -z "${_GMLAB_LD_LIBRARY_PATH_WAS_SET+x}" ]; then
    export _GMLAB_LD_LIBRARY_PATH_WAS_SET="${LD_LIBRARY_PATH+x}"
    export _GMLAB_LD_LIBRARY_PATH_PREVIOUS="${LD_LIBRARY_PATH-}"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
