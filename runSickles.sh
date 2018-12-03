rm -r runs
./copyToRosecrans.sh
ssh jmck746@rosecrans.cs.miami.edu << EOF
./runSicklesFromRosecrans.sh
EOF

#./getRunsFromRosecrans.sh # gets logs...but gets them too early because the above thing times out.
