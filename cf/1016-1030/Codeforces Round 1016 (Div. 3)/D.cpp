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
    int n, q;
    string op;
    cin >> n >> q;
    while (q--)
    {
        cin >> op;
        if (op[0] == '-')
        {
            int x, y;
            cin >> x >> y;
            int d = 1;
            int cnt = 1;
            x--, y--;
            while (x > 0 || y > 0)
            {
                if (x % 2 && y % 2)
                    d += cnt;
                else if (x % 2)
                    d += 2 * cnt;
                else if (y % 2)
                    d += 3 * cnt;
                cnt *= 4;
                x /= 2, y /= 2;
            }
            if (x == 1 && y == 1)
                d += 1;
            else if (x == 1)
                d += 2;
            else if (y == 1)
                d += 3;
            cout << d << endl;
        }
        else
        {
            int d;
            cin >> d;
            d--;
            int x = 1, y = 1;
            int cnt = 1;
            while (d > 0)
            {
                int now = d % 4;
                if (now == 1 || now == 2)
                    x += cnt;
                if (now == 3 || now == 1)
                    y += cnt;
                cnt *= 2;
                d /= 4;
            }
            cout << x << " " << y << endl;
        }
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