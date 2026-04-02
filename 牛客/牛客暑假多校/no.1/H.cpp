#include <bits/stdc++.h>
#define ff(i, a, b) for (int i = (a); i <= (b); i++)
#define ffr(i, a, b) for (int i = (a); i >= (b); i--)
#define mp make_pair
#define pii pair<int, int>
#define vc vector
#define vi vector<int>
#define vvi vector<vi>
using namespace std;
#ifdef __linux__
#define gc getchar_unlocked
#define pc putchar_unlocked
#else
#define gc _getchar_nolock
#define pc _putchar_nolock
#endif
inline bool blank(const char x) {return !(x^32)||!(x^10)||!(x^13)||!(x^9);}
template<typename Tp> inline void read(Tp &x) {x=0; register bool z=true; register char a=gc(); for(;!isdigit(a);a=gc()) if(a=='-') z=false; for(;isdigit(a);a=gc()) x=(x<<1)+(x<<3)+(a^48); x=(z?x:~x+1);}
inline void read(double &x) {x=0.0; register bool z=true; register double y=0.1; register char a=gc(); for(;!isdigit(a);a=gc()) if(a=='-') z=false; for(;isdigit(a);a=gc()) x=x*10+(a^48); if(a!='.') return x=z?x:-x,void(); for(a=gc();isdigit(a);a=gc(),y/=10) x+=y*(a^48); x=(z?x:-x);}
inline void read(char &x) {for(x=gc();blank(x)&&(x^-1);x=gc());}
inline void read(char *x) {register char a=gc(); for(;blank(a)&&(a^-1);a=gc()); for(;!blank(a)&&(a^-1);a=gc()) *x++=a; *x=0;}
inline void read(string &x) {x=""; register char a=gc(); for(;blank(a)&&(a^-1);a=gc()); for(;!blank(a)&&(a^-1);a=gc()) x+=a;}
template<typename T,typename ...Tp> inline void read(T &x,Tp &...y) {read(x),read(y...);}
template<typename Tp> inline void write(Tp x) {if(!x) return pc(48),void(); if(x<0) pc('-'),x=~x+1; register int len=0; register char tmp[64]; for(;x;x/=10) tmp[++len]=x%10+48; while(len) pc(tmp[len--]);}
inline void write(const double x) {register int a=6; register double b=x,c=b; if(b<0) pc('-'),b=-b,c=-c; register double y=5*powl(10,-a-1); b+=y,c+=y; register int len=0; register char tmp[64]; if(b<1) pc(48); else for(;b>=1;b/=10) tmp[++len]=floor(b)-floor(b/10)*10+48; while(len) pc(tmp[len--]); pc('.'); for(c*=10;a;a--,c*=10) pc(floor(c)-floor(c/10)*10+48);}
inline void write(const pair<int,double>x) {register int a=x.first; if(a<7) {register double b=x.second,c=b; if(b<0) pc('-'),b=-b,c=-c; register double y=5*powl(10,-a-1); b+=y,c+=y; register int len=0; register char tmp[64]; if(b<1) pc(48); else for(;b>=1;b/=10) tmp[++len]=floor(b)-floor(b/10)*10+48; while(len) pc(tmp[len--]); a&&(pc('.')); for(c*=10;a;a--,c*=10) pc(floor(c)-floor(c/10)*10+48);} else cout<<fixed<<setprecision(a)<<x.second;}
inline void write(const char x) {pc(x);}
inline void write(const bool x) {pc(x?49:48);}
inline void write(char *x) {fputs(x,stdout);}
inline void write(const char *x) {fputs(x,stdout);}
inline void write(const string &x) {fputs(x.c_str(),stdout);}
template<typename T,typename ...Tp> inline void write(T x,Tp ...y) {write(x),write(y...);}
void Murasame()
{
    int n, q;
    int cnt = 0, ans = 0, nowa = 0, nowb = 0;
    read(n, q);
    string s;
    read(s);
    s = '1' + s;
    vi st(n + 2, 0);
    while (q--)
    {
        int ty;
        read(ty);
        if (ty == 1)
        {
            int l, r;
            read(l, r);
            st[l]++;
            st[r + 1]--;
        }
        else
        {
            int l, a, b;
            read(l, a, b);
            cnt = 0, ans = 0, nowa = 0, nowb = 0;
            ff(i, 1, a - 1)
            {
                nowa += st[i];
            }
            ff(i, 1, b - 1)
            {
                nowb += st[i];
            }
            ff(i, 1, l)
            {
                nowa += st[a + i - 1];
                nowb += st[b + i - 1];
                if ((s[a + i - 1] - '0') ^ (nowa & 1) == (s[b + i - 1] - '0') ^ (nowb & 1))
                    cnt++;
                else
                {
                    ans += cnt * (cnt + 1) / 2;
                    cnt = 0;
                }
            }
            ans += cnt * (cnt + 1) / 2;
            write(ans);
            write('\n');
            
        }
    }
}
signed main()
{

    int _T = 1;
    // cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}