// Exact column-module membership in (Z/9Z)^24. Five-bit lanes hold residues.
#pragma once
struct Nine {
 W a=0,b=0;
 bool operator==(const Nine&) const = default;
 int at(int i)const{return int(((i<12?a:b)>>(5*(i%12)))&31);}
 void set(int i,int x){assert(x>=0&&x<9);W &w=i<12?a:b;int k=5*(i%12);w=(w&~(31ULL<<k))|(W(x)<<k);}
};
constexpr W ones9(){W x=0;for(int i=0;i<12;i++)x|=1ULL<<(5*i);return x;}
constexpr W NINE_ONES=ones9(),NINE_HIGH=16*NINE_ONES;
W reduce9(W x){return x-9*(((x+7*NINE_ONES)&NINE_HIGH)>>4);}
Nine add9(Nine x,Nine y){return {reduce9(x.a+y.a),reduce9(x.b+y.b)};}
Nine neg9(Nine x){return {reduce9(9*NINE_ONES-x.a),reduce9(9*NINE_ONES-x.b)};}
Nine mul9(Nine x,int k){assert(k>=0&&k<9);Nine r{};while(k){if(k&1)r=add9(r,x);k>>=1;if(k)x=add9(x,x);}return r;}
int unit_index(Nine x){for(int j=23;j>=0;j--)if(x.at(j)%3)return j;return -1;}
W div3_9(Nine x){W r=0;for(int j=0;j<24;j++){int t=x.at(j);assert(t%3==0);if(t)r|=1ULL<<(j+24*(t/3-1));}return r;}
Nine times3_9(W x){Nine r;for(int j=0;j<24;j++)r.set(j,3*int((x>>j&1)+2*(x>>(j+24)&1)));return r;}
Nine quotient9(const V&a,const V&b){Nine x;for(int j=0;j<24;j++){int d=int(a[j])-b[j];assert(d%3==0);x.set(j,(d/3%9+9)%9);}return x;}
struct Module9{
 // Chronological pivots: every later column vanishes at earlier pivots.
 array<Nine,24> unit{};array<int,24> pivot{};int n=0;
 array<W,24> torsion{};
 Nine reduce(Nine x) const {for(int i=0;i<n;i++){int k=x.at(pivot[i]);if(k)x=add9(x,mul9(unit[i],9-k));}return x;}
 void insert(Nine x){x=reduce(x);int j=unit_index(x);if(j<0){ternary_insert(torsion,div3_9(x));return;}
  constexpr int inverse[9]={0,1,5,0,7,2,0,4,8};assert(n<24);unit[n]=mul9(x,inverse[x.at(j)]);pivot[n++]=j;assert(unit[n-1].at(j)==1);
 }
 // Reproject early pure torsion columns after all unit pivots are known.
 array<W,24> reduced_torsion() const {array<W,24> p{};for(W t:torsion)if(t)ternary_insert(p,div3_9(reduce(times3_9(t))));return p;}
 bool contains(Nine x) const {x=reduce(x);if(unit_index(x)>=0)return false;return ternary_contains(reduced_torsion(),div3_9(x));}
 vector<Nine> generators() const {vector<Nine> g(unit.begin(),unit.begin()+n);for(W x:reduced_torsion())if(x)g.push_back(times3_9(x));return g;}
};
struct Lift27{V ref{};vector<Nine> basis;};
Lift27 make_lift27(const vector<Group>&g){assert(!g.empty()&&!g[0].vals.empty());Lift27 s;s.ref=g[0].vals[0];Module9 m;for(auto&x:g)for(auto&v:x.vals)m.insert(quotient9(v,s.ref));s.basis=m.generators();return s;}
bool possible27(const array<const Lift27*,4>&a,const V&target){V r=target;for(auto s:a)r=subv(r,s->ref);for(int j=0;j<24;j++)if(r[j]%3)return false;Module9 m;for(auto s:a)for(auto v:s->basis)m.insert(v);return m.contains(quotient9(r,V{}));}
