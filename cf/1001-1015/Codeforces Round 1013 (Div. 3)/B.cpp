#include <bits/stdc++.h>
// Finish Time: 2025/3/25 22:55:42 WA
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define vc vector
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
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
    int n, x;
    cin >> n >> x;
    vector<int> a(n, 0);
    ff(i, 0, n - 1)
            cin >>
        a[i];
    sort(all(a), greater<int>());
    int ans = 0;
    int minn = 1e9;
    int cnt = 0;
    for (int i = 0; i < n; i++)
    {
        if (a[i] >= x)
        {
            ans++;
        }
        else
        {
            cnt++;
            minn = min(minn, a[i]);
            if (minn * cnt >= x)
            {
                ans++;
                cnt = 0;
                minn = 1e9;
            }
        }
    }
    if (minn * cnt >= x)
    {
        ans++;
        cnt = 0;
        minn = 1e9;
    }
    cout << ans << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    //
    cin >> _T;
    while (_T--)
    {
        solve();
    }
    return 0;
}