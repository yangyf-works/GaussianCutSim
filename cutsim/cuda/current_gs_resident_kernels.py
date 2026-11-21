"""GPU BVH traversal and current-GS interval reconstruction, in Float64.

Uses the finite clipped-disc distance primitive from current_gs_surface.
No per-ray hit cap; hit buffers are allocated from a count/scan pass.
"""
RESIDENT_CUDA = r'''
#define WINDOW_OWNER (-9223372036854775807LL-1LL)
__device__ unsigned int expand_bits(unsigned int v){
 v=(v*0x00010001u)&0xFF0000FFu;v=(v*0x00000101u)&0x0F00F00Fu;
 v=(v*0x00000011u)&0xC30C30C3u;v=(v*0x00000005u)&0x49249249u;return v;
}
extern "C" __global__ void resident_codes(const double* g,long long n,const double* lo,const double* hi,unsigned int* codes){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;unsigned int q[3];
 for(int a=0;a<3;a++)q[a]=(unsigned int)fmin(1023.,fmax(0.,1023.*(g[19*i+a]-lo[a])/fmax(hi[a]-lo[a],1e-30)));
 codes[i]=(expand_bits(q[0])<<2)|(expand_bits(q[1])<<1)|expand_bits(q[2]);
}
extern "C" __global__ void resident_leaves(const double* g,const long long* order,long long n,long long leaves,double* boxes){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=leaves)return;
 double* b=boxes+6*(leaves+i);for(int a=0;a<3;a++){
  b[a]=i<n?g[19*order[i]+a]-g[19*order[i]+12]:INFINITY;
  b[a+3]=i<n?g[19*order[i]+a]+g[19*order[i]+12]:-INFINITY;}
}
extern "C" __global__ void resident_parents(double* boxes,long long begin,long long count){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;long long node=begin+i;
 for(int a=0;a<3;a++){boxes[6*node+a]=fmin(boxes[12*node+a],boxes[12*node+6+a]);
 boxes[6*node+a+3]=fmax(boxes[12*node+a+3],boxes[12*node+6+a+3]);}
}
__device__ double box_distance(const double* p,const double* b){
 double sum=0;for(int a=0;a<3;a++){double x=fmax(0.,fmax(b[a]-p[a],p[a]-b[a+3]));sum+=x*x;}return sum;
}
__device__ bool box_ray(const double* o,const double* d,double low,double high,const double* b){
 // Padding leaves have (+inf,-inf) bounds. The slab min/max below would
 // otherwise turn these inverted bounds into an infinite HIT for oblique
 // rays, traversing every padded descendant despite no geometry being there.
 if(b[0]>b[3])return false;
 for(int a=0;a<3;a++){
  if(d[a]==0){if(o[a]<b[a]-1e-10||o[a]>b[a+3]+1e-10)return false;}
  else {double x=(b[a]-1e-10-o[a])/d[a],y=(b[a+3]+1e-10-o[a])/d[a];low=fmax(low,fmin(x,y));high=fmin(high,fmax(x,y));if(low>high)return false;}}
 return true;
}
__device__ bool disc_hit(const double* o,const double* d,double low,double high,const double* g,double* t,double* sign,double* weight){
 double slope=dot(d,g+9);if(fabs(slope)<1e-10)return false;
 double delta[3]={g[0]-o[0],g[1]-o[1],g[2]-o[2]};*t=dot(delta,g+9)/slope;
 if(*t<low||*t>high)return false;
 double p[3]={o[0]+*t*d[0]-g[0],o[1]+*t*d[1]-g[1],o[2]+*t*d[2]-g[2]};
 double x=dot(p,g+3),y=dot(p,g+6),r=g[12];if(x*x+y*y>r*r+1e-12)return false;
 if(g[13]*x+g[14]*y+g[15]<-1e-12||g[16]*x+g[17]*y+g[18]<-1e-12)return false;
 *sign=slope>0?1.:-1.;*weight=fabs(slope)*exp(-2*(x*x+y*y)/(r*r));return true;
}
__device__ long long nearest_bvh(const double* q,const double* g,const double* boxes,const long long* order,
 long long n,long long leaves,double* value,long long* stats){
 int stack[64],top=0;stack[top++]=1;double best=INFINITY;long long chosen=-1,visited=0;*value=INFINITY;
 while(top){int node=stack[--top];if(box_distance(q,boxes+6*node)>best*best+1e-20)continue;
  if(node>=leaves){long long i=node-leaves;if(i>=n)continue;long long row=order[i];double h;visited++;
   double dd=dist(q,g+19*row,&h);if(dd<best||(dd==best&&(chosen<0||row<chosen))){best=dd;chosen=row;*value=dd*(h<0?-1.:1.);}}
  else {int a=node*2,b=a+1;double da=box_distance(q,boxes+6*a),db=box_distance(q,boxes+6*b);
   if(da<db){stack[top++]=b;stack[top++]=a;}else{stack[top++]=a;stack[top++]=b;}}}
 atomicAdd((unsigned long long*)(stats+0),1ULL);atomicAdd((unsigned long long*)(stats+1),(unsigned long long)visited);return chosen;
}
// Sort key of a hit: (distance, negative original source row - 1).
__device__ bool earlier(double t,long long row,double best,long long chosen){return t<best||(t==best&&row>chosen);}
__device__ long long first_segment(const double* o,const double* end,const double* g,const double* boxes,
 const long long* order,const long long* source,long long n,long long leaves,long long* stats){
 double d[3]={end[0]-o[0],end[1]-o[1],end[2]-o[2]},length=sqrt(dot(d,d));if(length<=1e-12||!n)return -1;
 for(int a=0;a<3;a++)d[a]/=length;
 double first[2]={INFINITY,INFINITY};long long rows[2]={-1,-1};long long visited=0;
 int stack[64],top=0;stack[top++]=1;
 while(top){int node=stack[--top];if(!box_ray(o,d,0.,length,boxes+6*node))continue;
  if(node<leaves){stack[top++]=node*2+1;stack[top++]=node*2;continue;}
  long long i=node-leaves;if(i>=n)continue;long long row=order[i];double t,sg,w;visited++;
  if(disc_hit(o,d,0.,length,g+19*row,&t,&sg,&w)){int side=sg>0?1:0;if(earlier(t,source[row],first[side],rows[side])){first[side]=t;rows[side]=source[row];}}}
 if(rows[0]<0&&rows[1]<0){atomicAdd((unsigned long long*)(stats+2),(unsigned long long)visited);return -1;}
 int side=earlier(first[0],rows[0],first[1],rows[1])?0:1;
 double bestw=-1.,bestt=INFINITY;long long chosen=-1,chosen_source=-1;top=0;stack[top++]=1;
 double stop=fmin(length,first[1-side]);
 while(top){int node=stack[--top];if(!box_ray(o,d,0.,stop,boxes+6*node))continue;
  if(node<leaves){stack[top++]=node*2+1;stack[top++]=node*2;continue;}
  long long i=node-leaves;if(i>=n)continue;long long row=order[i];double t,sg,w;visited++;
  if(disc_hit(o,d,0.,length,g+19*row,&t,&sg,&w)&&(sg>0?1:0)==side&&earlier(t,source[row],first[1-side],rows[1-side])){
   if(w>bestw||(w==bestw&&earlier(t,source[row],bestt,chosen_source))){bestw=w;bestt=t;chosen=row;chosen_source=source[row];}}}
 atomicAdd((unsigned long long*)(stats+2),(unsigned long long)visited);return chosen;
}
__device__ long long material_value(const double* p,const double* g,const double* boxes,const long long* order,
 const long long* source,long long n,long long leaves,double* value,long long* stats,int endpoint){
 long long row=nearest_bvh(p,g,boxes,order,n,leaves,value,stats);if(row<0||!isfinite(*value))return row;
 const double* q=g+19*row;double delta[3]={p[0]-q[0],p[1]-q[1],p[2]-q[2]},h=dot(delta,q+9);
 if(fabs(*value)<=fabs(h)+1e-7)return row;
 int base=endpoint?8:3;atomicAdd((unsigned long long*)(stats+base),1ULL);
 double end[3]={p[0],p[1],fmin(p[2]-1e-6,boxes[6+2]-1e-6)};
 long long hit=first_segment(p,end,g,boxes,order,source,n,leaves,stats);if(hit<0)return row;
 bool before=*value<0,inside=g[19*hit+11]<0;
 if(inside&&!before){end[2]=fmax(p[2]+1e-6,boxes[6+5]+1e-6);
  long long upper=first_segment(p,end,g,boxes,order,source,n,leaves,stats);
  inside=upper>=0&&g[19*upper+11]>0;atomicAdd((unsigned long long*)(stats+6),1ULL);
  if(!inside)atomicAdd((unsigned long long*)(stats+7),1ULL);}
 atomicAdd((unsigned long long*)(stats+base+1),1ULL);
 if(inside!=before)atomicAdd((unsigned long long*)(stats+base+2),1ULL);
 *value=fabs(*value)*(inside?-1.:1.);return row;
}
extern "C" __global__ void resident_query(const double* p,long long count,const double* g,const double* boxes,
 const long long* order,const long long* source,long long n,long long leaves,int mode,double* values,long long* owners,long long* stats){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;double value;
 long long row=mode?material_value(p+3*i,g,boxes,order,source,n,leaves,&value,stats,mode==2):nearest_bvh(p+3*i,g,boxes,order,n,leaves,&value,stats);
 values[i]=value;owners[i]=row>=0?source[row]:-1;
}
extern "C" __global__ void resident_segments(const double* p,const double* end,long long count,const double* g,const double* boxes,
 const long long* order,const long long* source,long long n,long long leaves,long long* owners,long long* stats){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 long long row=first_segment(p+3*i,end+3*i,g,boxes,order,source,n,leaves,stats);owners[i]=row>=0?source[row]:-1;
}
// Positive material evidence for rays with no generation witness. Nearest
// plane signs alone extend an open finite patch into an infinite half-space.
// Return the nearest *actual*, clipped-disc crossing in a cardinal direction.
// Equal-distance entry/exit conflicts and points on a face are rejected.
__device__ int enclosure_exit(const double* p,int axis,int side,const double* g,const double* boxes,
 const long long* order,long long n,long long leaves,long long* stats,double* distance){
 double d[3]={0,0,0};d[axis]=(double)side;
 double best=INFINITY;int result=0,stack[64],top=0;long long visited=0;stack[top++]=1;
 while(top){int node=stack[--top];if(!box_ray(p,d,0.,best+1e-8,boxes+6*node))continue;
  if(node<leaves){stack[top++]=node*2+1;stack[top++]=node*2;continue;}
  long long i=node-leaves;if(i>=n)continue;double t,sg,w;visited++;
  if(!disc_hit(p,d,0.,best+1e-8,g+19*order[i],&t,&sg,&w))continue;
  int state=(t>1e-8&&sg>0)?1:-1;
  if(t<best-1e-8){best=t;result=state;}
  else {best=fmin(best,t);if(state<0)result=-1;}
 }
 *distance=best;atomicAdd((unsigned long long*)(stats+2),(unsigned long long)visited);return result;
}
extern "C" __global__ void resident_enclosure(const double* p,long long count,const double* g,const double* boxes,
 const long long* order,const long long* source,long long n,long long leaves,signed char* evidence,long long* stats){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 evidence[i]=0;if(!n)return;
 // Missing surfaces are unknown, never implicit solid. Near clipped curved
 // seams different directions can disagree. Prefer the nearer boundary
 // evidence, but an exit is evidence only with an opposing exit as well.
 // A closer entry/contact always vetoes a distant enclosing pair.
 double solid=INFINITY,air=INFINITY;
 for(int j=0;j<3;j++){int axis=(j+2)%3;
  double da,db;
  int a=enclosure_exit(p+3*i,axis,-1,g,boxes,order,n,leaves,stats,&da);
  int b=enclosure_exit(p+3*i,axis,1,g,boxes,order,n,leaves,stats,&db);
  if(a<0)air=fmin(air,da);if(b<0)air=fmin(air,db);
  if(a>0&&b>0)solid=fmin(solid,fmin(da,db));
 }
 evidence[i]=(solid<air-1e-8)?1:(isfinite(air)?-1:0);
}
// Candidate selection only: no clip planes/radius limits are generated here.
// Probe the proposed GS support against finite BEFORE-state surfaces. Ignore
// co-oriented patches of the same continuous sheet, including nearer ones
// that could otherwise hide a real foreign boundary farther along a probe.
extern "C" __global__ void resident_boundary_risk(const double* proposed,const long long* ids,long long count,
 const double* g,const double* boxes,const long long* order,const long long* source,long long n,long long leaves,
 const long long* old_ids,unsigned char* risk,long long* stats){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 const double* p=proposed+13*i;risk[i]=0;long long visited=0;
 if(!n||p[12]<=0)return;
 for(int angle=0;angle<16&&!risk[i];angle++){
  double a=angle*0.3926990816987241548,c=cos(a),s=sin(a),d[3];
  for(int k=0;k<3;k++)d[k]=c*p[3+k]+s*p[6+k];
  int stack[64],top=0;stack[top++]=1;
  while(top&&!risk[i]){
   int node=stack[--top];if(!box_ray(p,d,0.,p[12],boxes+6*node))continue;
   if(node<leaves){stack[top++]=2*node+1;stack[top++]=2*node;continue;}
   long long leaf=node-leaves;if(leaf>=n)continue;long long row=order[leaf];const double* q=g+19*row;visited++;
   if(ids[i]==old_ids[row]&&dot(p+9,q+9)>0.)continue;
   if(dot(d,q+9)<=1e-10)continue;
   double t,sign,weight;if(disc_hit(p,d,0.,p[12],q,&t,&sign,&weight))risk[i]=1;
  }
 }
 atomicAdd((unsigned long long*)(stats+2),(unsigned long long)visited);
}
extern "C" __global__ void resident_hits(const double* origins,const double* directions,const double* domain,long long count,
 const double* g,const double* boxes,const long long* order,const long long* source,long long n,long long leaves,
 const long long* offsets,long long* counts,double* output,int write,long long* stats){
 long long ray=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(ray>=count)return;
 const double* o=origins+3*ray;const double* d=directions+3*ray;double low=domain[2*ray],high=domain[2*ray+1];
 int stack[64],top=0;stack[top++]=1;long long found=0,visited=0;
 while(top){int node=stack[--top];if(!box_ray(o,d,low,high,boxes+6*node))continue;
  if(node<leaves){stack[top++]=2*node+1;stack[top++]=2*node;continue;}
  long long i=node-leaves;if(i>=n)continue;long long row=order[i];double t,sign,w;visited++;
  if(disc_hit(o,d,low,high,g+19*row,&t,&sign,&w)){
   if(write){double* out=output+5*(offsets[ray]+found);out[0]=(double)ray;out[1]=t;out[2]=sign;out[3]=(double)(-source[row]-1);out[4]=w;}found++;}}
 if(!write)counts[ray]=found;
 atomicAdd((unsigned long long*)(stats+2),(unsigned long long)visited);
}
extern "C" __global__ void resident_intervals(const double* origins,const double* directions,const double* domain,long long rays,
 const double* g,const double* boxes,const long long* order,const long long* source,long long n,long long leaves,
 const double* hits,const long long* hit_offsets,const long long* offsets,long long* counts,double* ends,long long* owners,
 long long* summary,long long* stats){
 long long ray=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(ray>=rays)return;
 double low=domain[2*ray],high=domain[2*ray+1],entry=low;long long entry_owner=WINDOW_OWNER,written=0;
 long long begin=hit_offsets[ray],stop=hit_offsets[ray+1],merged=0;bool inside=false;
 if(begin==stop){
  double p[3],tip[3];for(int a=0;a<3;a++){p[a]=origins[3*ray+a]+low*directions[3*ray+a];tip[a]=origins[3*ray+a]+high*directions[3*ray+a];}
  double seed,end;material_value(p,g,boxes,order,source,n,leaves,&seed,stats,0);
  material_value(tip,g,boxes,order,source,n,leaves,&end,stats,0);inside=seed<0;
  if((seed<0)!=(end<0)){
   double lo=low,hi=high;long long owner=-1;
   for(int iteration=0;iteration<24;iteration++){
    double mid=(lo+hi)*.5;for(int a=0;a<3;a++)p[a]=origins[3*ray+a]+mid*directions[3*ray+a];
    double value;long long row=material_value(p,g,boxes,order,source,n,leaves,&value,stats,0);owner=row>=0?-source[row]-1:0;
    if((value<0)==(seed<0))lo=mid;else hi=mid;}
   double t=(lo+hi)*.5;
   if(inside){long long j=offsets[ray]+written++;ends[2*j]=low;ends[2*j+1]=t;owners[2*j]=WINDOW_OWNER;owners[2*j+1]=owner;inside=false;}
   else {entry=t;entry_owner=owner;inside=true;}
   atomicAdd((unsigned long long*)(summary+1),1ULL);
  }
 }else {
  inside=hits[5*begin+2]>0;
  for(long long i=begin;i<stop;){long long j=i+1,best=i;
   while(j<stop&&hits[5*j+2]==hits[5*i+2]){if(hits[5*j+4]>hits[5*best+4])best=j;j++;}
   merged+=j-i-1;double t=hits[5*best+1];long long who=(long long)hits[5*best+3];
   if(hits[5*best+2]<0){if(!inside){entry=t;entry_owner=who;}inside=true;}
   else {if(inside&&entry<t){long long k=offsets[ray]+written++;ends[2*k]=entry;ends[2*k+1]=t;owners[2*k]=entry_owner;owners[2*k+1]=who;}inside=false;}i=j;}
 }
 if(inside&&entry<high){long long j=offsets[ray]+written++;ends[2*j]=entry;ends[2*j+1]=high;owners[2*j]=entry_owner;owners[2*j+1]=WINDOW_OWNER;}
 counts[ray]=written;atomicAdd((unsigned long long*)(summary+0),(unsigned long long)merged);
}
'''
