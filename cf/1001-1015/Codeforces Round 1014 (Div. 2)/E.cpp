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
int mod = 1e9 + 7;
int qpow(int a, int b)
{
    int ans = 1;
    a %= mod;
    while (b)
    {
        if (b & 1)
            ans = (ans * a) % mod;
        a = (a * a) % mod;
        b >>= 1;
    }
    return ans;
}
void solve()
{
    int n, m, k;
    cin >> n >> m >> k;
    int cnt = 0, sum = 0;
    ff(i, 1, k)
    {
        int x, y, z;
        cin >> x >> y >> z;
        if ((x == 1 && y == 1) || (x == n && y == m) || (x == 1 && y == m) || (x == n && y == 1))
            continue;
        if (x == 1 || x == n || y == 1 || y == m)
        {
            cnt++;
            sum += z;
        }
    }
    if (cnt == 2 * (n + m - 4))
    {
        cout << (sum % 2 ? 0 : qpow(2, n * m - k)) << endl;
    }
    else
    {
        cout << qpow(2, n * m - k - 1) << endl;
    }
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