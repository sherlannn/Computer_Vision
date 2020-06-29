```
+ Edge Detection
```
Edge detection using convolution and :</br>
1. Caddy mask</br></br>
2. Laplacian mask:</br>
([0 -1 0]</br>
,[-1 4 -1]</br>
,[0 -1 0])</br></br>
3.Sobel mask:</br>
3.1: Sobel x :</br>
([-1,0,1],</br>
,[-2,0,2],</br>
,[-1,0,1])</br>
3.2: Soble y :</br>
([1 2 1],</br>
,[0 0 0]</br>
,[-1 -2 -1])</br></br>
4.Robert mask:</br>
4.1: Robert x:</br>
([0 0 0]</br>
,[0 1 0]</br>
,[0 0 -1])</br>
4.2: Robert y:</br>
([0 0 0]</br>
,[0 0 1]</br>
,[0 -1 0])</br></br>
You can see the results:</br></br>
![](result.PNG)
