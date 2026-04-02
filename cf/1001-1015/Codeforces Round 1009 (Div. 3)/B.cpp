#include <bits/stdc++.h>
// Finish Time: 2025/3/12 13:59:33 AC
#define int long long
#define i128 __int128
#define ld long double
#define ull unsigned long long
#define ff(x,y,z) for(int(x)=(y);(x)<=(z);++(x))
#define ffg(x,y,z) for(int(x)=(y);(x)>=(z);--(x))
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
#define inf 0x7fffffff
#define endl '\n' 
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
void solve()
{
    int n;
    cin >> n;
    vector<int> a(n);
    for(int i = 0; i < n; i++)
        cin >> a[i];
        if(n==1)
        {
            cout<<a[0]<<endl;
            return;
        }
        int ans=a[0]+a[1]-1;
    for(int i = 2; i < n; i++)
    ans=ans+a[i]-1;
    cout<<ans<<endl;
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