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
    int n, k;
    cin >> n >> k;
    vi a(n + 1), b(n + 1);
    int maxa = -inf, mina = inf;
    ff(i, 1, n)
    {
        cin >> a[i];
        maxa = max(maxa, a[i]);
        mina = min(mina, a[i]);
    }
    map<int, int> mp;
    int bb = -1;
    ff(i, 1, n)
    {
        cin >> b[i];
        if (b[i] != -1)
        {
            mp[b[i] + a[i]]++;
            bb = b[i] + a[i];
        }
    }
    if (mp.size() > 1)
    {
        cout << 0 << endl;
    }
    else if (mp.size() == 1)
    {
        ff(i, 1, n)
        {
            if (a[i] > bb || a[i] + k < bb)
            {
                cout << 0 << endl;
                return;
            }
        }
        cout << 1 << endl;
    }
    else
    {
        cout << max((mina + k) - maxa + 1, 0LL) << endl;
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
        Murasame();
    }
    return 0;
}