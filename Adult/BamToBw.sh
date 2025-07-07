
InputFold="/lustre/home/zhangfy/fetalmonkey/atac/06.merge/ADR/"
OutputFold="./ATAC/"
SampleName="ATAC-RM06050A_E80-ADR"

bamToBed -i ${InputFold}/${SampleName}.bam > ${OutputFold}/${SampleName}.bed
/lustre/home/zhangfy/Pipeline/Anaconda3/envs/py38/bin/macs2 callpeak -p 0.01 -t  ${OutputFold}/${SampleName}.bed -g 3083880790 -B --keep-dup all --nomodel --shift -100 --extsize 200 -n ${SampleName}/ --outdir  ${OutputFold}/${SampleName}/
/lustre/home/zhangfy/Pipeline/Anaconda3/envs/py38/bin/macs2 bdgcmp -t ${OutputFold}/${SampleName}/${SampleName}_treat_pileup.bdg -c ${OutputFold}/${SampleName}/${SampleName}_control_lambda.bdg --o-prefix ${OutputFold}/${SampleName}/${SampleName} -m FE
bedtools slop -i ${OutputFold}/${SampleName}/${SampleName}_FE.bdg -g /lustre/home/zhangfy/fetalmonkey/Macaque.t-to-t/macaque_chromSize.txt -b 0 | bedClip stdin /lustre/home/zhangfy/fetalmonkey/Macaque.t-to-t/macaque_chromSize.txt ${OutputFold}/${SampleName}/${SampleName}.fc.signal.bedgraph
sort -k1,1 -k2,2n ${OutputFold}/${SampleName}/${SampleName}.fc.signal.bedgraph | awk 'BEGIN{{OFS="\\t"}}{{if (NR==1 || NR>1 && (prev_chr!=$1 || prev_chr==$1 && prev_chr_e<=$2)) {{print $0}}; prev_chr=$1; prev_chr_e=$3;}}' > ${OutputFold}/${SampleName}/${SampleName}.fc.signal.sort.bedgraph
bedGraphToBigWig ${OutputFold}/${SampleName}/${SampleName}.fc.signal.sort.bedgraph /lustre/home/zhangfy/fetalmonkey/Macaque.t-to-t/macaque_chromSize.txt ${OutputFold}/${SampleName}/${SampleName}.fc.signal.bw
reads_num=`cat ${OutputFold}/${SampleName}.bed|wc -l`
sval=`awk -v num="$reads_num" 'BEGIN {print num / 1000000}'`
/lustre/home/zhangfy/Pipeline/Anaconda3/envs/py38/bin/macs2 bdgcmp -t ${OutputFold}/${SampleName}/${SampleName}_treat_pileup.bdg -c ${OutputFold}/${SampleName}/${SampleName}_control_lambda.bdg --o-prefix ${OutputFold}/${SampleName}/${SampleName} -m ppois -S "$sval" 
bedtools slop -i ${OutputFold}/${SampleName}/${SampleName}_ppois.bdg -g /lustre/home/zhangfy/fetalmonkey/Macaque.t-to-t/macaque_chromSize.txt -b 0 | bedClip stdin /lustre/home/zhangfy/fetalmonkey/Macaque.t-to-t/macaque_chromSize.txt ${OutputFold}/${SampleName}/${SampleName}.pval.signal.bedgraph
sort -k1,1 -k2,2n ${OutputFold}/${SampleName}/${SampleName}.pval.signal.bedgraph | awk 'BEGIN{{OFS="\\t"}}{{if (NR==1 || NR>1 && (prev_chr!=$1 || prev_chr==$1 && prev_chr_e<=$2)) {{print $0}}; prev_chr=$1; prev_chr_e=$3;}}' > ${OutputFold}/${SampleName}/${SampleName}.pval.signal.sort.bedgraph
bedGraphToBigWig ${OutputFold}/${SampleName}/${SampleName}.pval.signal.sort.bedgraph /lustre/home/zhangfy/fetalmonkey/Macaque.t-to-t/macaque_chromSize.txt ${OutputFold}/${SampleName}/${SampleName}.pval.signal.bw


InputFold="/lustre/home/zhangfy/fetalmonkey/atac/06.merge/ADR/"
OutputFold="./RNA/"
SampleName="ATAC-RM06050A_E80-ADR"

/lustre/home/zhangfy/Pipeline/bin/STAR --runMode inputAlignmentsFromBAM --inputBAMfile ${InputFold}/${SampleName}Aligned.sortedByCoord.out.bam --outWigType bedGraph --outWigStrand Unstranded --outFileNamePrefix ${OutputFold}/${SampleName}

/lustre/home/zhangfy/Pipeline/bin/bedSort Signal.UniqueMultiple.str1.out.bg Signal.UniqueMultiple.str1.out.sorted.bg
/lustre/home/zhangfy/Pipeline/bin/bedGraphToBigWig Signal.UniqueMultiple.str1.out.sorted.bg /lustre/home/zhangfy/Pipeline/bin/Macaca.chrom.sizes ${sample}_minusAll.bw
/lustre/home/zhangfy/Pipeline/bin/bedSort Signal.UniqueMultiple.str2.out.bg Signal.UniqueMultiple.str2.out.sorted.bg
/lustre/home/zhangfy/Pipeline/bin/bedGraphToBigWig Signal.UniqueMultiple.str2.out.sorted.bg /lustre/home/zhangfy/Pipeline/bin/Macaca.chrom.sizes ${sample}_plusAll.bw
