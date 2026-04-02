#include <bits/stdc++.h>
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
    int n, t, x;
    cin >> n >> t >> x;
    int temp = (n - x) / x;
    int need = 1 + (temp - 1) / 2 + (temp % 2 == 0 && temp * x + x != n);
    if (need <= t)
        cout << n << " ";
    else
    {
        int ans = x;
        if ((t - 1) * 2 + 1 <= temp)
            ans += x * ((t - 1) * 2 + 1);
        else
        ans=n;
        cout << ans << " ";
    }
    if (t == 0 || x > 1)
        cout << x << endl;
    else
        cout << 2 << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    // cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}