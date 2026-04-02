#include <bits/stdc++.h>
#define int long long
#define i128 __int128
#define ld long double
#define ull unsigned long long
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
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
    vector<int> num(n + 1), hx(1000006, 0), pre(1000006, 0), vis(1000006, 0);
    int tt = 0;
    for (int i = 1; i <= n; i++)
    {
        cin >> num[i];
        if (num[i] == 0)
            continue;
        if (hx[num[i]] == 0)
            tt++;
        hx[num[i]]++;
    }
    int now = 0;
    int ans = 0;
    for (int i = 1; i <= n; i++)
    {
        if (num[i] == 0)
        {
            now = i;
            continue;
        }
        hx[num[i]]--;
        if (hx[num[i]] == 0)
            tt--;
        if (pre[num[i]])
        {
            if (pre[num[i]] < now && vis[num[i]] == 0)
            {
                ans += tt;
                vis[num[i]] = 1;
            }
        }
        else
            pre[num[i]] = i;
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