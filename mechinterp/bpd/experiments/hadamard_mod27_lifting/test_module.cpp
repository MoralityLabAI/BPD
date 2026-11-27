#define HAD17_LIBRARY
#include "module_join.cpp"
#include <random>
// Independent scalar membership via row and column elimination over Z/9.
using Scalar=array<int,24>;
int mod9(int x){return (x%9+9)%9;}
Scalar decode(Nine x){Scalar r{};for(int j=0;j<24;j++)r[j]=x.at(j);return r;}
bool scalar_member(vector<Scalar> cols,Scalar y){
 int r=0,k=0; int nc=cols.size(); constexpr int inv[9]={0,1,5,0,7,2,0,4,8};
 // Diagonalize units by elementary row and column operations; transform y by row ops.
 while(r<24&&k<nc){int a=-1,b=-1;for(int j=k;j<nc&&a<0;j++)for(int i=r;i<24;i++)if(cols[j][i]%3){a=i;b=j;break;}if(a<0)break;
  swap(cols[k],cols[b]); for(auto &v:cols)swap(v[r],v[a]);swap(y[r],y[a]);int d=inv[cols[k][r]];
  for(auto &v:cols)v[r]=mod9(v[r]*d);y[r]=mod9(y[r]*d);
  for(int i=r+1;i<24;i++){int f=cols[k][i];for(auto &v:cols)v[i]=mod9(v[i]-f*v[r]);y[i]=mod9(y[i]-f*y[r]);}
  for(int j=k+1;j<nc;j++){int f=cols[j][r];for(int i=r;i<24;i++)cols[j][i]=mod9(cols[j][i]-f*cols[k][i]);}
  ++r;++k;
 }
 // The residual block is all multiples of three; divide and eliminate over F3.
 for(int i=r;i<24;i++){if(y[i]%3)return false;y[i]/=3;for(int j=k;j<nc;j++){assert(cols[j][i]%3==0);cols[j][i]/=3;}}
 while(r<24&&k<nc){int a=-1,b=-1;for(int j=k;j<nc&&a<0;j++)for(int i=r;i<24;i++)if(cols[j][i]){a=i;b=j;break;}if(a<0)break;
  swap(cols[k],cols[b]);for(int j=k;j<nc;j++)swap(cols[j][r],cols[j][a]);swap(y[r],y[a]);int d=cols[k][r];for(int j=k;j<nc;j++)cols[j][r]=cols[j][r]*d%3;y[r]=y[r]*d%3;
  for(int i=r+1;i<24;i++){int f=cols[k][i];for(int j=k;j<nc;j++)cols[j][i]=(cols[j][i]+6-f*cols[j][r])%3;y[i]=(y[i]+6-f*y[r])%3;}
  ++r;++k;
 }
 return all_of(y.begin()+r,y.end(),[](int x){return x==0;});
}
Nine encode(const Scalar&v){Nine x;for(int j=0;j<24;j++)x.set(j,v[j]);return x;}
// Enumerate all coefficient tuples for at most four columns, ordinary scalar arithmetic.
bool brute(const vector<Scalar>&c,Scalar y,int i=0){if(i==int(c.size()))return all_of(y.begin(),y.end(),[](int x){return x==0;});for(int k=0;k<9;k++){if(brute(c,y,i+1))return true;for(int j=0;j<24;j++)y[j]=mod9(y[j]-c[i][j]);}return false;}
int main(){mt19937 rng(87701);int positives=0,negatives=0,stronger=0,brutes=0;W checks=0;
 for(int it=0;it<3000;it++){
  int n=it<512?1+rng()%4:1+rng()%32;int d=it<512?1+rng()%8:1+rng()%24;
  vector<Scalar> cols;Module9 m;Scalar y{};
  for(int i=0;i<n;i++){Scalar x{};for(int j=0;j<d;j++)x[j]=(it%5==0?3*(rng()%3):rng()%9);cols.push_back(x);m.insert(encode(x));int coef=rng()%9;for(int j=0;j<24;j++)y[j]=mod9(y[j]+coef*x[j]);}
  if(it%2)y[rng()%d]=mod9(y[rng()%d]+(it%3?3:1)); // arbitrary negative candidate, classified by oracle
  Nine t=encode(y);bool a=m.contains(t),b=scalar_member(cols,y);assert(a==b);checks++;
  if(it<512){assert(a==brute(cols,y));checks++;brutes++;}
  Module9 compact;for(Nine g:m.generators())compact.insert(g);assert(compact.contains(t)==a);checks++;
  // Every original column remains in compact module, including torsion before unit pivot.
  for(auto &v:cols){assert(compact.contains(encode(v)));checks++;}
  if(!(it%2))assert(a);positives+=a;negatives+=!a;
  array<W,24> p{};V z{};for(auto &x:cols){V v{};for(int j=0;j<24;j++)v[j]=3*x[j];ternary_insert(p,quotient3(v,z));}V target{};for(int j=0;j<24;j++)target[j]=3*y[j];bool mod3=ternary_contains(p,quotient3(target,z));assert(!a||mod3);checks++;stronger+=mod3&&!a;
  Nine u=encode(cols.front()),v=encode(cols.back());for(int j=0;j<24;j++){assert(add9(u,v).at(j)==(u.at(j)+v.at(j))%9);assert(neg9(u).at(j)==mod9(-u.at(j)));checks+=2;}
 }
 // Explicit nonfree module: 3 is in the span of 3, 1 is not. New unit pivot after torsion.
 Module9 m;Nine x{};x.set(0,3);m.insert(x);assert(m.contains(x));Nine y{};y.set(0,1);assert(!m.contains(y));y.set(1,1);m.insert(y);assert(m.contains(x)&&m.contains(y));Nine z{};z.set(1,3);assert(m.contains(z));z.set(1,1);assert(!m.contains(z));
 cout<<"tests=3000 positives="<<positives<<" negatives="<<negatives<<" stronger_than_mod9="<<stronger<<" brute_controls="<<brutes<<" assertions="<<checks+5<<'\n';
}
