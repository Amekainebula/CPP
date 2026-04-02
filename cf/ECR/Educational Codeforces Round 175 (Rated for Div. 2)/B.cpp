#include <bits/stdc++.h>
// Finish Time: 2025/2/28 13:55:56 AC
#define int long long
#define i128 __int128
#define ld long double
#define ull unsigned long long
#define lowbit(x) (x & -x)
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define rall(x) x.rbegin(), x.rend()
#define sz(x) (int)(x).size()
#define INF 0x7fffffffffffffff
#define endl '\n' 
using namespace std;
void solve()
{
    int n,x,k;
    cin >> n >> x >> k;
    int cntr=0,cntl=0;
    string s;
    cin >> s;
    int cnt=0;
    int ans=0;
    for(int i=0;i<n&&k>0&&x!=0;i++)
    {
        if(s[i]=='R')x++;
        else x--;
        k--;
        if(x==0)
        {
            ans++;
            break;
        }
        
        
    }
    if(ans!=0)
    {
        bool flag=false;
        for(int i=0;i<n;i++)
        {
           
            if(s[i]=='R')
            cntr++;
            else cntl++;
            cnt++;
            if(cntr==cntl&&cntr!=0&&cntl!=0)
            {
                flag=true;
                break;
            }
        }
        if(flag)
        ans+=k/cnt;
    }
    cout << ans << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    cin >> T;
    while (T--)
    {
        solve();
    }
    return 0;
}