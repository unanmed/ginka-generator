for i in {$1...$2}
do
    sh gan.sh "$i"
    echo "第 $i 次循环完成"
done
