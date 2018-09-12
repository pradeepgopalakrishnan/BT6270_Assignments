for i in $(seq 0.06 0.002 0.3)
do
	octave ./HHmodel_solution.m $i
done
