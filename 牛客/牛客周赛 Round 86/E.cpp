#include <bits/stdc++.h>
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
const int N = 100 * 100 + 10;
void solve()
{
    int n;
    cin >> n;
    vector<int> a(n);
    ff(i, 0, n - 1)
        cin >> a[i];
    sort(all(a));
    vector<int> sum(n + 1, 0);
    ff(i, 0, n - 1)
        sum[i + 1] = sum[i] + a[i];
    int min_ = inf;
    ff(i, 0, n - 1)
    {
        int val = a[i];
        int pre = sum[i];
        if (pre <= val)
            continue;
        bitset<N> dp;
        dp[0] = 1;
        ff(k, 0, i - 1)
        {
            int x = a[k];
            dp |= dp << x;
        }
        auto p = dp._Find_next(val);
        if (p != dp.size() && p <= pre)
        {
            min_ = min(min_, (int)p + val);
        }
    }
    cout << (min_ == inf? -1 : min_) << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    cin >> _T;
    while (_T--)
    {
        solve();
    }
    return 0;
}