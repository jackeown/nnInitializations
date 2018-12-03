ssh jmck746@rosecrans.cs.miami.edu << EOF
rm -r classes/deepLearning/OrthoOrdent/*
EOF

scp -r * jmck746@rosecrans.cs.miami.edu:~/classes/deepLearning/OrthoOrdent
