import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt

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
    heigh=length(mask(arange(),1))
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

    kernel = np.ones((70,70),np.uint8)
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
    if (logical_not(isempty(upper_bound))):
        upper_bound=upper_bound.T
        a=copy(upper_bound)
        lins=copy(a)
        dev=20
        if (length(upper_bound(arange(),1)) == 3):
            dist1=sqrt((lins(1,1) - lins(2,1)) ** 2 + (lins(1,2) - lins(2,2)) ** 2)
            dist2=sqrt((lins(3,1) - lins(2,1)) ** 2 + (lins(3,2) - lins(2,2)) ** 2)
            a1=(lins(1,2) - lins(2,2)) / (lins(1,1) - lins(2,1))
            a2=(lins(2,2) - lins(3,2)) / (lins(2,1) - lins(3,1))
            lower_flg=0
            if (logical_not(isempty(lower_bound))):
                b=copy(lower_bound)
                blins=copy(b)
                if (length(lower_bound(arange(),1)) == 3):
                    lower_flg=1
                    if (abs(a1) - abs(a2) > 0):
                        case_in=1
                        flag=0
                        para_lines=(blins(1,2) - blins(2,2)) / (blins(1,1) - blins(2,1))
                        per_lines=(lins(2,2) - lins(1,2)) / (lins(2,1) - lins(1,1))
                        para_lines1=concat([blins(1,1),blins(1,2),blins(2,1),blins(2,2)])
                        per_lines1=concat([lins(1,1),lins(1,2),lins(2,1),lins(2,2)])
                    else:
                        flag=0
                        case_in=2
                        para_lines=(blins(3,2) - blins(2,2)) / (blins(3,1) - blins(2,1))
                        per_lines=(lins(2,2) - lins(3,2)) / (lins(2,1) - lins(3,1))
                        para_lines1=concat([blins(3,1),blins(3,2),blins(2,1),blins(2,2)])
                        per_lines1=concat([lins(3,1),lins(3,2),lins(2,1),lins(2,2)])
            if (lower_flg != 1):
                flag=2
                if (abs(a1) - abs(a2) > 0):
                    if (dist1 > 350):
                        case_in=3
                        flag=0
                        para_lines=(lins(3,2) - lins(2,2)) / (lins(3,1) - lins(2,1))
                        per_lines=(lins(2,2) - lins(1,2)) / (lins(2,1) - lins(1,1))
                        para_lines1=concat([lins(3,1),lins(3,2),lins(2,1),lins(2,2)])
                        per_lines1=concat([lins(1,1),lins(1,2),lins(2,1),lins(2,2)])
                    else:
                        dev=30
                        case_in=4
                        rot_dev=- 45
                        per_lines=(lins(3,2) - lins(2,2)) / (lins(3,1) - lins(2,1))
                        per_lines1=concat([lins(3,1),lins(3,2),lins(2,1),lins(2,2)])
                        flag=1
                else:
                    if (dist2 > 350):
                        case_in=5
                        flag=0
                        para_lines=(lins(2,2) - lins(1,2)) / (lins(2,1) - lins(1,1))
                        per_lines=(lins(3,2) - lins(2,2)) / (lins(3,1) - lins(2,1))
                        para_lines1=concat([lins(2,1),lins(2,2),lins(1,1),lins(1,2)])
                        per_lines1=concat([lins(3,1),lins(3,2),lins(2,1),lins(2,2)])
                    else:
                        dev=30
                        case_in=6
                        rot_dev=50
                        per_lines=(lins(2,2) - lins(1,2)) / (lins(2,1) - lins(1,1))
                        per_lines1=concat([lins(1,1),lins(1,2),lins(2,1),lins(2,2)])
                        flag=1
        if (length(upper_bound(arange(),1)) == 2):
            b=copy(lower_bound)
            blins=copy(b)
            if (logical_not(isempty(lower_bound))):
                if (length(lower_bound(arange(),1)) == logical_or(2,length(lower_bound(arange(),1))) == 3):
                    c1=(dot(2,heigh) - blins(1,2) - blins(2,2)) / 2
                    c2=(lins(1,2) + lins(2,2)) / 2
                    if (length(lower_bound(arange(),1)) == 2):
                        if (c1 > c2):
                            if ((dot(2,heigh) - blins(1,2) - blins(2,2)) / 2 > 10):
                                per_lines=(blins(1,2) - blins(2,2)) / (blins(1,1) - blins(2,1))
                                per_lines1=concat([blins(1,1),blins(1,2),blins(2,1),blins(2,2)])
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
                                per_lines1=concat([lins(1,1),lins(1,2),lins(2,1),lins(2,2)])
                            else:
                                flag=2
                                case_in=14
                    else:
                        flag=2
                        case_in=14
                    if (length(lower_bound(arange(),1)) == 3):
                        b1=(blins(1,2) - blins(2,2)) / (blins(1,1) - blins(2,1))
                        b2=(blins(2,2) - blins(3,2)) / (blins(2,1) - blins(3,1))
                        flag=1
                        if (abs(b1) - abs(b2) > 0):
                            case_in=8
                            per_lines=(blins(1,2) - blins(2,2)) / (blins(1,1) - blins(2,1))
                            per_lines1=concat([blins(1,1),blins(1,2),blins(2,1),blins(2,2)])
                            dev=30
                            rot_dev=- 40
                        else:
                            case_in=9
                            per_lines=(blins(2,2) - blins(3,2)) / (blins(2,1) - blins(3,1))
                            per_lines1=concat([blins(3,1),blins(3,2),blins(2,1),blins(2,2)])
                            dev=30
                            rot_dev=40
            else:
                flag=1
                case_in=10
                per_lines=(lins(2,2) - lins(1,2)) / (lins(2,1) - lins(1,1))
                per_lines1=concat([lins(1,1),lins(1,2),lins(2,1),lins(2,2)])
    else:
        if (logical_not(isempty(lower_bound))):
            flag=1
            b=copy(lower_bound)
            blins=copy(b)
            if (length(lower_bound(arange(),1)) == 2):
                per_lines=(blins(1,2) - blins(2,2)) / (blins(1,1) - blins(2,1))
                per_lines1=concat([blins(1,1),blins(1,2),blins(2,1),blins(2,2)])
                case_in=11
            if (length(lower_bound(arange(),1)) == 3):
                b1=(blins(1,2) - blins(2,2)) / (blins(1,1) - blins(2,1))
                b2=(blins(2,2) - blins(3,2)) / (blins(2,1) - blins(3,1))
                if (abs(b1) > abs(b2)):
                    case_in=12
                    per_lines=(blins(1,2) - blins(2,2)) / (blins(1,1) - blins(2,1))
                    per_lines1=concat([blins(1,1),blins(1,2),blins(2,1),blins(2,2)])
                    dev=30
                    rot_dev=- 40
                else:
                    case_in=13
                    per_lines=(blins(2,2) - blins(3,2)) / (blins(2,1) - blins(3,1))
                    per_lines1=concat([blins(3,1),blins(3,2),blins(2,1),blins(2,2)])
                    dev=30
                    rot_dev=40
        else:
            flag=2
            case_in=14
    case_in

    label_i=bwlabel(mask)
    measurements=regionprops(label_i,'Area','Perimeter','BoundingBox')
    allAreas=concat([measurements.Area])
    allPerims=concat([measurements.Perimeter])
    allbounds=cat(1,measurements.BoundingBox)
    a=[]
    if (logical_not(isempty(allbounds))):
        max_b=max(allbounds(arange(),3))
        max_h=max(allbounds(arange(),4))
        for i in arange(1,length(allAreas)).reshape(-1):
            b=allbounds(i,3)
            c=allbounds(i,4)
            if (logical_and(((allAreas(i) / allPerims(i)) < logical_or(logical_or(logical_or(2,(b > logical_and(200,(allAreas(i) / allPerims(i))) < 6)),(c > logical_and(200,(allAreas(i) / allPerims(i))) < 6)),b) > logical_or(500,c) > 500),allAreas(i)) > 30):
                a=concat([a,i])
        keeper=ismember(label_i,a)
        mask=copy(keeper)

        rm_small=imopen(mask,strel('disk',7))

        maskt=mask - rm_small
        label_i=bwlabel(maskt)
        measurements=regionprops(label_i,'BoundingBox')
        allbounds=cat(1,measurements.BoundingBox)
        max_b1=max(allbounds(arange(),3))
        max_h1=max(allbounds(arange(),4))
        max_h
        max_h1
        if (abs(max_h - max_h1) > 300):
            rm_small=imopen(mask,strel('disk',10))
            maskt=mask - rm_small
        mask=copy(maskt)
        label_i=bwlabel(mask)
        measurements=regionprops(label_i,'Area','Perimeter','BoundingBox')
        allAreas=concat([measurements.Area])
        allPerims=concat([measurements.Perimeter])
        allbounds=cat(1,measurements.BoundingBox)
        a=[]
        for i in arange(1,length(allAreas)).reshape(-1):
            b=allbounds(i,3)
            c=allbounds(i,4)
            if (logical_and(((allAreas(i) / allPerims(i)) < logical_or(logical_or(logical_or(6,(b > logical_and(200,(allAreas(i) / allPerims(i))) < 6)),(c > logical_and(200,(allAreas(i) / allPerims(i))) < 6)),b) > logical_or(500,c) > 500),allAreas(i)) > 100):
                a=concat([a,i])
        keeper=ismember(label_i,a)
        mask=copy(keeper)
    case_final=0
    mask=imdilate(mask,strel('disk',1))
    Bimage=copy(mask)
    if (sum(sum(Bimage)) > 10000):
        if (case_in == logical_or(7,case_in) == logical_or(10,case_in) == 11):
            line_per,case_final=check_yardline(Bimage,case_in,per_lines,per_lines1,nargout=2)
            if (case_final != 0):
                case_final,line_yax,line_xax,fin,magni=dual_lines(Bimage,line_per,line_per,per_lines1,per_lines1,case_in,case_final,nargout=5)
        if (case_in == logical_or(4,case_in) == logical_or(6,case_in) == logical_or(9,case_in) == logical_or(8,case_in) == logical_or(12,case_in) == 13):
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
                line_len=copy(line_len1)
                magni=copy(magni1)
                line_xax=concat([line_per1(fin1,1),line_per1(fin1,2),line_per1(fin1,3),line_per1(fin1,4)])
            else:
                line_len=copy(line_len2)
                magni=copy(magni2)
                line_xax=concat([line_per2(fin2,1),line_per2(fin2,2),line_per2(fin2,3),line_per2(fin2,4)])
    return case_final,line_yax,line_xax,magni
if __name__ == '__main__':
    pass
