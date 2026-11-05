# Restore exactly the library search path present before activation.
if [ "${_GMLAB_LD_LIBRARY_PATH_WAS_SET+x}" = x ]; then
    if [ "${_GMLAB_LD_LIBRARY_PATH_WAS_SET}" = x ]; then
        export LD_LIBRARY_PATH="${_GMLAB_LD_LIBRARY_PATH_PREVIOUS}"
    else
        unset LD_LIBRARY_PATH
    fi
    unset _GMLAB_LD_LIBRARY_PATH_WAS_SET _GMLAB_LD_LIBRARY_PATH_PREVIOUS
fi
