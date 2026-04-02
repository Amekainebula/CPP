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
#define vi vector<int>
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
void Murasame()
{
    int n, m, l, r;
    cin >> n >> m >> l >> r;
    if (n == m)
    {
        cout << l << ' ' << r << endl;
        return;
    }
    int cnt = n - m;
    if (l < 0)
    {
        int t = min(-l, cnt);
        cnt -= t;
        l += t;
    }
    if (r > 0 && cnt > 0)
    {
        int t = min(r, cnt);
        cnt -= t;
        r -= t;
    }
    cout << l << ' ' << r << endl;
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