// Exact enumeration of inversion-symmetric support tuples on Z23.
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <map>
#include <vector>
#include <bit>
using namespace std;
int main(){
 constexpr unsigned q=23,n=4096;
 array<uint32_t,n> expanded{}; array<unsigned,n> mass{},sq{};
 for(unsigned x=0;x<n;x++){
  for(unsigned j=0;j<12;j++)if(x>>j&1){expanded[x]|=1u<<j; if(j)expanded[x]|=1u<<(q-j);}
  mass[x]=popcount(expanded[x]);
  for(unsigned j=0;j<q;j++)if(expanded[x]>>j&1){unsigned k=2*j%q; if(k<12)sq[x]|=1u<<k;}
 }
 vector<uint16_t> mul(n*n);
 for(unsigned s=0;s<n;s++){
  array<unsigned,12> col{};
  for(unsigned j=0;j<12;j++){
   auto rot=[&](unsigned t){return ((expanded[s]<<t)|(expanded[s]>>(q-t)))&((1u<<q)-1);};
   col[j]=(j?rot(j)^rot(q-j):expanded[s])&(n-1);
  }
  for(unsigned w=1;w<n;w++)mul[s*n+w]=mul[s*n+(w&(w-1))]^col[countr_zero(w)];
  assert(mul[s*n+s]==sq[s]);
 }
 map<array<unsigned,4>,uint64_t> hist; uint64_t count=0;
 for(unsigned s=0;s<n;s++)for(unsigned w=0;w<n;w++){
  unsigned t=1^s^w^sq[s]^sq[w]^mul[s*n+w];
  array<unsigned,4> k={mass[t],mass[t^s],mass[t^w],mass[t^s^w^1]};
  assert((k[0]%2+k[1]%2+k[2]%2+k[3]%2)==3);
  unsigned m=k[0]+k[1]+k[2]+k[3];
  if(m%12!=11)continue;
  ++hist[k];++count;
 }
 cerr<<"masks="<<count<<" histogram="<<hist.size()<<"\n";
 for(auto [k,c]:hist)cout<<k[0]<<","<<k[1]<<","<<k[2]<<","<<k[3]<<","<<c<<"\n";
}
