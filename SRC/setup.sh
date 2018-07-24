#!/bin/bash

rm -rf Match-2.3.1
echo "*** Getting codes ***"
wget -q -O - 'http://www.lorraine-lisiecki.com/match-2.3.1.tgz' | tar xvzf - | awk '{print " ",$0}'
(
	cd Match-2.3.1/ || exit
	echo "*** Patching (using patch for gcc version 6.3, change to 4.9 if you have issues) ***"

	## cat ../patch.gcc-4.9 | patch -p1 | awk '{print " ",$0}'
	cat ../patch.gcc-6.3 | patch -p1 | awk '{print " ",$0}'
	
	cd Match/ || exit
	echo "*** Compiling ***"
	make clean | awk '{print " ",$0}'
	make | awk '{print " ",$0}'
	[ -f match ] && echo "File will be installed on 'match'"

	echo ""
	echo "*** Done ***"
	echo ""
	cp -v match ../../../
	cd ../Examples/
	cp LR04core b806l ../../../
)
rm -rf Match-2.3.1


