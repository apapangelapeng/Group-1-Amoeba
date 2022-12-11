RESULT_FILE="result_1_2.cvs"
HEADER="teeth_length,teeth_gap,density,metabolism,start_size,index,succeeded,turns_taken,final_size,seperation_error"
echo "$HEADER">>$RESULT_FILE
TL=2
TG=1
size=(3 5 8 15 25)
density=(0.01 0.05 0.1 0.2)
seed=(5 2 1 1)
metabolism=(0.05 0.1 0.25 0.4 1.0)






for s in ${!size[@]}; do
    for m in ${!metabolism[@]}; do
        for d in ${!density[@]}; do
        ##density and seed are correlated
         
            echo "running size ${size[$s]}, density ${density[$d]}, metabolism ${metabolism[$m]}, teeth gap $TG, teeth length $TL"
            python3 main.py -p 1 -A ${size[$s]} -d ${density[$d]} -m ${metabolism[$m]} -s ${seed[$m]} -tg $TG -tl $TL -nv -ng
                
        done
    done
done
    






""
