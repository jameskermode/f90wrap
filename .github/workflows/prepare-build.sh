echo "prepare-build.sh received environment ARCH=${ARCH} RUNNER_OS=${RUNNER_OS}"


if [ "${RUNNER_OS}" == "macOS" ]; then

    # taken from https://github.com/MacPython/gfortran-install/blob/master/gfortran_utils.sh#L97
    function install_arm64_cross_gfortran {
	curl -L -O https://github.com/isuruf/gcc/releases/download/gcc-10-arm-20210228/gfortran-darwin-arm64.tar.gz
	export GFORTRAN_SHA=f26990f6f08e19b2ec150b9da9d59bd0558261dd
	if [[ "$(shasum gfortran-darwin-arm64.tar.gz)" != "${GFORTRAN_SHA}  gfortran-darwin-arm64.tar.gz" ]]; then
            echo "shasum mismatch for gfortran-darwin-arm64"
            exit 1
	fi
	sudo mkdir -p /opt/
	sudo cp "gfortran-darwin-arm64.tar.gz" /opt/gfortran-darwin-arm64.tar.gz
	pushd /opt
        sudo tar -xvf gfortran-darwin-arm64.tar.gz
        sudo rm gfortran-darwin-arm64.tar.gz
	popd
	export FC_ARM64="$(find /opt/gfortran-darwin-arm64/bin -name "*-gfortran")"
	local libgfortran="$(find /opt/gfortran-darwin-arm64/lib -name libgfortran.dylib)"
	local libdir=$(dirname $libgfortran)

	export FC_ARM64_LDFLAGS="-L$libdir -Wl,-rpath,$libdir"
	if [[ "$ARCH" == "arm64" ]]; then
        export FC=$FC_ARM64
	    export F90=$FC
	    export F95=$FC
	    export F77=$FC
	fi
    }

    if [ "$ARCH" == "arm64" ]; then
		echo "Installing arm64 cross compiler..."
		install_arm64_cross_gfortran
    fi
fi

# Python build dependencies
pip install oldest-supported-numpy
