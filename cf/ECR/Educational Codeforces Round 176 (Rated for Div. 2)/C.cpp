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
    int n, m;
    cin >> n >> m;
    vector<int> a(m + 1);
    ff(i, 1, m) cin >> a[i];
    sort(a.begin() + 1, a.end());
    vector<int> arr(m + 2, 0);
    ff(i, 1, m) arr[i] = arr[i - 1] + a[i];
    arr[m + 1] = arr[m];
    int ans = 0;
    ff(i, 1, m)
    {
        if (a[i] >= n - 1)
        {
            ans += (m - i) * (m + 1 - i) / 2 * (n - 1);
            break;
        }
        else
        {
            int l = lower_bound(a.begin() + i + 1, a.end(), n - a[i]) - a.begin();
            int r = lower_bound(a.begin() + i + 1, a.end(), n - 1) - a.begin();
            int cnt = r - l;
            ans += (arr[r - 1] - arr[l - 1]) + a[i] * cnt - n * cnt + cnt;
            if (r != m + 1)
                ans += a[i] * (m - r + 1);
        }
    }
    cout << ans * 2 << endl;
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