import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt
import math
from skimage.measure import label, regionprops

def find_inner_boundaries(mask=None,im=None,upper_bound=None,lower_bound=None,*args,**kwargs):
    pq=(im)
    case_in=0
    lower_bound=lower_bound.T
    flag=10
    magni=0
    line_yax=0
    line_xax=0
    #sasi
    mask=im2double(mask)
    #Thresholding
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.GaussianBlur(mask,(5,5),0)
    ret2,mask = cv2.threshold(mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # thresh=graythresh(mask)
    # mask=im2bw(mask,thresh)
    #
    heigh=len(mask[:,1])
    #
    #Breaking into Connected Components
    nb_components,output,stats,centroids = cv2.connectedComponentsWithStats(mask,connectivity = 8)
    sizes = stats[1:,-1]

    #Components with area in range (10000 and 5000000) is taken only such that complete audience is removed
    mask = np.zeros((output.shape))
    for i in range(0,nb_components-1):
        if(sizes[i] >= 10000 and sizes[i] <= 5000000):
            mask[ output == i+1 ] = 255
    mask = mask.astype(np.uint8)
    # mask now has ground - players

    #removing players from ground
    kernel = np.ones((20,20),np.uint8)
    mask2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((50,50),np.uint8)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    mask2 = mask2.astype(np.uint8)
    ## mask2 now has just the ground
    # filt=bwareafilt(mask,concat([10000,5000000]))
    # temp=imopen(filt,strel('disk',20))
    # mask2=imclose(temp,strel('disk',50))
    mask = np.invert(mask)
    # mask=imcomplement(mask)
    mask = np.multiply(mask,mask2)
    # mask=multiply(mask,mask2)

    #
    kernel = np.ones((7,7),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # mask=imclose(mask,strel('disk',7))
    rot_dev=0
    if (len(upper_bound)!=0):
        upper_bound=upper_bound.T
        a=upper_bound
        lins=a
        dev=20
        if (len(upper_bound[:,1]) == 3):
            dist1=math.sqrt((lins(1,1) - lins(2,1)) ** 2 + (lins(1,2) - lins(2,2)) ** 2)
            dist2=math.sqrt((lins(3,1) - lins(2,1)) ** 2 + (lins(3,2) - lins(2,2)) ** 2)
            a1=(lins(1,2) - lins(2,2)) / (lins(1,1) - lins(2,1))
            a2=(lins(2,2) - lins(3,2)) / (lins(2,1) - lins(3,1))
            lower_flg=0
            if (len(lower_bound)!=0):
                b=lower_bound
                blins=b
                if (len(lower_bound[:,1]) == 3):
                    lower_flg=1
                    if (abs(a1) - abs(a2) > 0):
                        case_in=1
                        flag=0
                        para_lines=(blins(1,2) - blins(2,2)) / (blins(1,1) - blins(2,1))
                        per_lines=(lins(2,2) - lins(1,2)) / (lins(2,1) - lins(1,1))
                        para_lines1=np.concatenate([blins(1,1),blins(1,2),blins(2,1),blins(2,2)])
                        per_lines1=np.concatenate([lins(1,1),lins(1,2),lins(2,1),lins(2,2)])
                    else:
                        flag=0
                        case_in=2
                        para_lines=(blins(3,2) - blins(2,2)) / (blins(3,1) - blins(2,1))
                        per_lines=(lins(2,2) - lins(3,2)) / (lins(2,1) - lins(3,1))
                        para_lines1=np.concatenate([blins(3,1),blins(3,2),blins(2,1),blins(2,2)])
                        per_lines1=np.concatenate([lins(3,1),lins(3,2),lins(2,1),lins(2,2)])
            if (lower_flg != 1):
                flag=2
                if (abs(a1) - abs(a2) > 0):
                    if (dist1 > 350):
                        case_in=3
                        flag=0
                        para_lines=(lins(3,2) - lins(2,2)) / (lins(3,1) - lins(2,1))
                        per_lines=(lins(2,2) - lins(1,2)) / (lins(2,1) - lins(1,1))
                        para_lines1=np.concatenate([lins(3,1),lins(3,2),lins(2,1),lins(2,2)])
                        per_lines1=np.concatenate([lins(1,1),lins(1,2),lins(2,1),lins(2,2)])
                    else:
                        dev=30
                        case_in=4
                        rot_dev=- 45
                        per_lines=(lins(3,2) - lins(2,2)) / (lins(3,1) - lins(2,1))
                        per_lines1=np.concatenate([lins(3,1),lins(3,2),lins(2,1),lins(2,2)])
                        flag=1
                else:
                    if (dist2 > 350):
                        case_in=5
                        flag=0
                        para_lines=(lins(2,2) - lins(1,2)) / (lins(2,1) - lins(1,1))
                        per_lines=(lins(3,2) - lins(2,2)) / (lins(3,1) - lins(2,1))
                        para_lines1=np.concatenate([lins(2,1),lins(2,2),lins(1,1),lins(1,2)])
                        per_lines1=np.concatenate([lins(3,1),lins(3,2),lins(2,1),lins(2,2)])
                    else:
                        dev=30
                        case_in=6
                        rot_dev=50
                        per_lines=(lins(2,2) - lins(1,2)) / (lins(2,1) - lins(1,1))
                        per_lines1=np.concatenate([lins(1,1),lins(1,2),lins(2,1),lins(2,2)])
                        flag=1
        if (len(upper_bound[:,1]) == 2):
            b=lower_bound
            blins=b
            if (len(lower_bound)!=0):
                if (len(lower_bound[:,1]) == 2 or len(lower_bound[:,1]) == 3):
                    c1=((2*heigh) - blins(1,2) - blins(2,2)) / 2
                    c2=(lins(1,2) + lins(2,2)) / 2
                    if (len(lower_bound[:,1]) == 2):
                        if (c1 > c2):
                            if (((2*heigh) - blins(1,2) - blins(2,2)) / 2 > 10):
                                per_lines=(blins(1,2) - blins(2,2)) / (blins(1,1) - blins(2,1))
                                per_lines1=np.concatenate([blins(1,1),blins(1,2),blins(2,1),blins(2,2)])
                                flag=1
                                case_in=7
                            else:
                                flag=2
                                case_in=14
                        else:
                            if ((lins(1,2) + lins(2,2)) / 2 > 10):
                                flag=1
                                case_in=10
                                per_lines=(lins(2,2) - lins(1,2)) / (lins(2,1) - lins(1,1))
                                per_lines1=np.concatenate([lins(1,1),lins(1,2),lins(2,1),lins(2,2)])
                            else:
                                flag=2
                                case_in=14
                    else:
                        flag=2
                        case_in=14
                    if (len(lower_bound[:,1]) == 3):
                        b1=(blins(1,2) - blins(2,2)) / (blins(1,1) - blins(2,1))
                        b2=(blins(2,2) - blins(3,2)) / (blins(2,1) - blins(3,1))
                        flag=1
                        if (abs(b1) - abs(b2) > 0):
                            case_in=8
                            per_lines=(blins(1,2) - blins(2,2)) / (blins(1,1) - blins(2,1))
                            per_lines1=np.concatenate([blins(1,1),blins(1,2),blins(2,1),blins(2,2)])
                            dev=30
                            rot_dev=- 40
                        else:
                            case_in=9
                            per_lines=(blins(2,2) - blins(3,2)) / (blins(2,1) - blins(3,1))
                            per_lines1=np.concatenate([blins(3,1),blins(3,2),blins(2,1),blins(2,2)])
                            dev=30
                            rot_dev=40
            else:
                flag=1
                case_in=10
                per_lines=(lins(2,2) - lins(1,2)) / (lins(2,1) - lins(1,1))
                per_lines1=np.concatenate([lins(1,1),lins(1,2),lins(2,1),lins(2,2)])
    else:
        if (len(lower_bound)!=0):
            flag=1
            b=lower_bound
            blins=b
            if (len(lower_bound[:,1]) == 2):
                per_lines=(blins(1,2) - blins(2,2)) / (blins(1,1) - blins(2,1))
                per_lines1=np.concatenate([blins(1,1),blins(1,2),blins(2,1),blins(2,2)])
                case_in=11
            if (len(lower_bound[:,1]) == 3):
                b1=(blins(1,2) - blins(2,2)) / (blins(1,1) - blins(2,1))
                b2=(blins(2,2) - blins(3,2)) / (blins(2,1) - blins(3,1))
                if (abs(b1) > abs(b2)):
                    case_in=12
                    per_lines=(blins(1,2) - blins(2,2)) / (blins(1,1) - blins(2,1))
                    per_lines1=np.concatenate([blins(1,1),blins(1,2),blins(2,1),blins(2,2)])
                    dev=30
                    rot_dev=- 40
                else:
                    case_in=13
                    per_lines=(blins(2,2) - blins(3,2)) / (blins(2,1) - blins(3,1))
                    per_lines1=np.concatenate([blins(3,1),blins(3,2),blins(2,1),blins(2,2)])
                    dev=30
                    rot_dev=40
        else:
            flag=2
            case_in=14
    case_in

    #label_i=bwlabel(mask)
    label_i=label(mask)
    #measurements=regionprops(label_i,'Area','Perimeter','BoundingBox')
    #
    allbounds=[]
    allAreas = []
    allPerims = []
    for region in regionprops(label_i):
        if region.area >= 100:
            minr, minc, maxr, maxc = region.bbox
            allbound = [minr, minc, maxr, maxc ]
            allbounds.append(allbound)
            allArea = region.area
            allAreas.append(allArea)
            allPerim = region.perimeter
            allPerims.append(allPerim)

    #
    #allAreas=[measurements.Area]
    #allPerims=[measurements.Perimeter]
    #allbounds=np.concatenate(1,measurements.BoundingBox)
    a=[]
    if (len(allbounds)!=0):
        max_b=max(allbounds[:,3])
        max_h=max(allbounds[:,4])
        for i in range(allAreas):
            b=allbounds[i,3]
            c=allbounds[i,4]
            div = allAreas[i] / allPerims[i]
            if (((div < 2 ) or (b>200 and div < 6) or (c > 200 and div<6) or b> 500 or c > 500) and allAreas[i]):
                a=np.concatenate([a,i])
        # keeper=ismember(label_i,a)
        if label_i in a:
            keeper = 1
        else:
            keeper = 0
        mask=keeper

        #rm_small=imopen(mask,strel('disk',7))
        kernel = np.ones((7,7),np.uint8)
        rm_small = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        maskt=mask - rm_small
        label_i=bwlabel(maskt)
        measurements=regionprops(label_i,'BoundingBox')
        allbounds=np.concatenate(1,measurements.BoundingBox)
        max_b1=max(allbounds[:,3])
        max_h1=max(allbounds[:,4])
        max_h
        max_h1
        if (abs(max_h - max_h1) > 300):
            # rm_small=imopen(mask,strel('disk',10))
            kernel = np.ones((10,10),np.uint8)
            rm_small = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            maskt=mask - rm_small
        mask=maskt
        label_i=bwlabel(mask)
        measurements=regionprops(label_i,'Area','Perimeter','BoundingBox')
        allAreas=[measurements.Area]
        allPerims=[measurements.Perimeter]
        allbounds=np.concatenate(1,measurements.BoundingBox)
        a=[]
        for i in range(allAreas):
            b=allbounds(i,3)
            c=allbounds(i,4)
            div = allAreas(i) / allPerims(i)
            if ((div<6 or (b>200 and div <6) or (c>200 and div<6) or b>500 or c>500) and allAreas[i]>100):
                a=np.concatenate([a,i])
        # keeper=ismember(label_i,a)
        if label_i in a:
            keeper = 1
        else:
            keeper = 0
        mask=keeper
        mask=keeper
    case_final=0
    #
    mask=imdilate(mask,strel('disk',1))
    #
    Bimage=mask
    if (sum(sum(Bimage)) > 10000):
        if (case_in==7 or case_in==10 or case_in==11):
            line_per,case_final=check_yardline(Bimage,case_in,per_lines,per_lines1,nargout=2)
            if (case_final != 0):
                case_final,line_yax,line_xax,fin,magni=dual_lines(Bimage,line_per,line_per,per_lines1,per_lines1,case_in,case_final,nargout=5)
        if (case_in==4 or case_in==6 or case_in==9 or case_in==8 or case_in==12 or case_in==13):
            line_per=case_perpendicular(Bimage,per_lines,rot_dev,dev)
            line_per=merge_lines(line_per)
            case_final,line_yax,line_xax,line_len,magni,fin=dual_lines(Bimage,line_per,line_per,per_lines1,per_lines1,case_in,case_final,nargout=6)
        if (flag == 0):
            dev=5
            line_per1,line_par=case_parallel(Bimage,per_lines,para_lines,dev,nargout=2)
            line_per1=merge_lines(line_per1)
            case_final,line_yax,line_xax,line_len1,magni1,fin1=dual_lines(Bimage,line_per1,line_par,per_lines1,para_lines1,case_in,case_final,nargout=6)
            dev=15
            line_per2,line_par=case_parallel(Bimage,per_lines,para_lines,dev,nargout=2)
            line_per2=merge_lines(line_per2)
            case_final,line_yax,line_xax,line_len2,magni2,fin2=dual_lines(Bimage,line_per2,line_par,per_lines1,para_lines1,case_in,case_final,nargout=6)
            if (line_len1 > line_len2):
                line_len=line_len1
                magni=magni1
                line_xax=np.concatenate([line_per1(fin1,1),line_per1(fin1,2),line_per1(fin1,3),line_per1(fin1,4)])
            else:
                line_len=line_len2
                magni=magni2
                line_xax=np.concatenate([line_per2(fin2,1),line_per2(fin2,2),line_per2(fin2,3),line_per2(fin2,4)])
    return case_final,line_yax,line_xax,magni
if __name__ == '__main__':
    pass
