#include <bits/stdc++.h>
// Finish Time: 2025/2/28 13:55:58 AC
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
    int n;
    cin >> n;
    int ans1=n/15,ans2=n%15;
    cout<<1+ans1*3+min(ans2,2*1LL)<<endl;
    
    
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