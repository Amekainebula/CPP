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
    int n, k;
    cin >> n >> k;
    vector<int> a(n), b(n + 1);
    ff(i, 0, n - 1)
    {
        cin >> a[i];
        b[i] = a[i];
    }
    sort(all(b), greater<int>());
    int ans = 0;
    if (k == 1)
    {
        int temp = -1;
        ff(i, 1, n - 2)
            temp = max(temp, a[i]);
        ans = max(max(a[0], a[n - 1]) + temp, a[0] + a[n - 1]);
    }
    else
    {
        ff(i, 0, k)
            ans += b[i];
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