gcc -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro build/temp.linux-i686-2.7/affine/extensions/C_extensions.o -o build/lib.linux-i686-2.7/affine/model/_C_extensions.so
cp build/lib.linux-i686-2.7/affine/model/_C_extensions.so /usr/local/lib/python2.7/dist-packages
