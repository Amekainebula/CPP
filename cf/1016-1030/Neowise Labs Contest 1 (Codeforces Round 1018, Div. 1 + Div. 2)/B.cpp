#include <bits/stdc++.h>
// Finish Time: 2025/4/20 14:00:46 AC
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define vc vector
#define vi vector<int>
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
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
void Murasame()
{
    int n, k;
    cin >> n >> k;
    vi a(n), b(n);
    int sum1 = 0, sum2 = 0;
    int ans = 0;
    ff(i, 0, n - 1)
    {
        cin >> a[i];
    }
    ff(i, 0, n - 1)
    {
        cin >> b[i];
    }
    vi ta(n), tb(n);
    ff(i, 0, n - 1)
    {
        ta[i] = max(a[i], b[i]);
        tb[i] = min(a[i], b[i]);
        ans += ta[i];
    }
    sort(all(tb), greater<int>());
    ff(i, 0, k - 2)
        ans += tb[i];
    cout << ans + 1 << endl;
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
        Murasame();
    }
    return 0;
}