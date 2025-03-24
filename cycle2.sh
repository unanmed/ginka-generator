start=$1
end=$2
for ((i=start; i<=end; i=i+1))
do
    sh gan.sh "$i"
    echo "第 $i 次循环完成"
done
