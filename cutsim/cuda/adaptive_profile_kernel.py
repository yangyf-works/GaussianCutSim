"""Fused, bounded tool-chart child-ray evaluation and local AABB clipping."""
KERNEL=r'''
__device__ double column_middle(double x,double r,double low,double high,int ball){
 return .5*(high+(ball?fmax(low,r-sqrt(fmax(0.,r*r-x*x))):low));
}
extern "C" __global__ void adaptive_children(const long long* parents,long long count,
 const double* origins,const double* dirs,const double* packed,const int* codes,
 const double* box,int divisions,double* output,unsigned char* valid){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 long long parent=parents[i];const double* a=packed+18*parent;double* out=output+18*i;
 double low=a[15],high=a[16];int ball=(int)a[17];
 int child=i%(divisions*divisions),kind=codes[parent];double old_s=(a[0]+a[1])*.5,old_q=(a[2]+a[3])*.5;
 double ds=(a[1]-a[0])/divisions,dq=(a[3]-a[2])/divisions;
 out[6]=a[0]+(child/divisions)*ds;out[7]=out[6]+ds;out[8]=a[2]+(child%divisions)*dq;out[9]=out[8]+dq;
 double s=(out[6]+out[7])*.5,q=(out[8]+out[9])*.5,r=a[4],t=a[6];
 double o[3],d[3],side[3]={-a[8],a[7],0};
 for(int k=0;k<3;k++){o[k]=origins[3*parent+k];d[k]=dirs[3*parent+k];out[15+k]=a[7+k];}
 double z=o[2]+t*d[2];
 if(kind==0){d[0]=sin(s/r)*cos(q);d[1]=sin(s/r)*sin(q);d[2]=-cos(s/r);}
 else if(kind==1){o[2]+=s-old_s;d[0]=cos(q);d[1]=sin(q);d[2]=0;}
 else if(kind==2||kind==3){double h=kind==2?low:high,m=column_middle(s,r,low,high,ball);
  o[0]+=s*cos(q)-old_s*cos(old_q);o[1]+=s*sin(q)-old_s*sin(old_q);o[2]=z-h+m;t=fabs(h-m);}
 else if(kind==4){for(int k=0;k<3;k++){o[k]+=(s-old_s)*a[7+k];d[k]=sin(q/r)*side[k];}d[2]=-cos(q/r);}
 else if(kind==5){for(int k=0;k<3;k++)o[k]+=(s-old_s)*a[7+k];o[2]+=q-old_q;}
 else {for(int k=0;k<3;k++)o[k]+=(s-old_s)*a[7+k]+(q-old_q)*side[k];
  double m=column_middle(q,r,low,high,ball);o[2]+=m-column_middle(old_q,r,low,high,ball);t=fabs((d[2]<0?low:high)-m);}
 double feed_dot=0;for(int k=0;k<3;k++)feed_dot+=(o[k]+t*d[k]-a[10+k])*a[7+k];
 bool good=a[14]==0||a[14]*feed_dot>=-1e-12;double lo=0,hi=t+a[13];
 for(int k=0;k<3;k++){
  if(d[k]==0){if(o[k]<box[k]||o[k]>box[k+3])good=false;}
  else{double x=(box[k]-o[k])/d[k],y=(box[k+3]-o[k])/d[k];lo=fmax(lo,fmin(x,y));hi=fmin(hi,fmax(x,y));}
  out[k]=o[k];out[3+k]=d[k];}
 out[10]=r;out[11]=a[5]/divisions;out[12]=t;out[13]=lo;out[14]=hi;valid[i]=good&&lo<hi;
}
'''
